import torch
import numpy as np
from sklearn import metrics
import time, sys
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import shap
from transformers import Trainer, TrainingArguments
from common.dataset import LightDataset


class TorchTrainer:
    def __init__(self, model, train_dataloader, valid_dataloader, optimizer, loss_fn
                 , device, accumulation_steps, log_func, vectorizer = None, test_dataloader = None):
      
      if model:
        self.model = model.to(device)
      self.device = device
      self.train_dataloader = train_dataloader
      self.valid_dataloader = valid_dataloader
      self.optimizer = optimizer
      self.loss_fn = loss_fn
      self.accumulation_steps = accumulation_steps
      self.loss_history = []
      self.log = log_func
      self.vectorizer = vectorizer
      self.test_dataloader= test_dataloader


    def __reset_weights(self):
      '''
        Try resetting model weights to avoid
        weight leakage.
      '''
      for layer in self.model.children():
        if hasattr(layer, 'reset_parameters'):
          self.log(f'Reset trainable parameters of layer = {layer}')
          layer.reset_parameters()


    def k_fold_fit(self, k_folds, dataset, batch_size, num_epochs, save_path = ""):
      '''
        K-fold training.
      '''
      
      # Define the K-fold Cross Validator
      kfold = KFold(n_splits=k_folds, shuffle=False)

      # K-fold Cross Validation model evaluation
      for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        # Print
        self.log(f'----------------- FOLD {fold} -----------------')
        
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
        
        # Define data loaders for training and testing data in this fold
        self.train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size
          , sampler=train_subsampler)
        self.valid_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size
          , sampler=val_subsampler)

        self.__reset_weights()
        
        self.fit(num_epochs= num_epochs, save_path = save_path, save_only_best_models= False
          , save_all_epochs=False, early_stopping= False)

        save_path += f'/model-fold-{fold}.pt'
        torch.save(self.model, save_path)
        self.log("The model for fold " + str(fold) + " was saved succssfully.")


    def fit(self, num_epochs, patience = 2, save_path = "", save_only_best_models= False
      , save_all_epochs= True, early_stopping= True):

      # Early stopping 
      the_last_loss = 100
      trigger_times = 0
      the_best_loss = 0.95
      # ----my code -----
      loss_log = []
      # ----my code -----
      
      # clear_output()
      valid_acc = 0

      for epoch in range(num_epochs):
        self.log(f"{'-'* 20} Epoch {epoch + 1} / {num_epochs} {'-'* 20}")
        start_time = time.time()
        train_loss = self.__train_model()
        the_current_loss = self.__valid_model()

        # ----my code -----
        loss_log.append(the_current_loss)
        # ----my code -----

        the_last_loss = the_current_loss

        if save_only_best_models:
          # Save the best model
          if the_current_loss < the_best_loss:
            torch.save(self.model, save_path + "/loss"+ str(the_current_loss.item()) +".pt")
            the_best_loss = the_current_loss
            self.log("The model with loss " + str(the_best_loss.item()) + " was saved succssfully.")              
        elif save_all_epochs:
          torch.save(self.model, save_path + "/epoch" + str(epoch + 1) + "_loss" + str(the_current_loss.item()) +".pt")
          self.log("The model with loss " + str(the_current_loss.item()) + " was saved succssfully.")

        time_elapsed = time.time() - start_time
        self.log('\n  Epoch complete in: %.0fm %.0fs \n' % (time_elapsed // 60, time_elapsed % 60))

        # Early stopping
        if the_current_loss > the_last_loss and early_stopping:
          trigger_times += 1
          self.log('trigger times:' + str(trigger_times))
          if trigger_times >= patience:
            self.log('Early stopping!\nStart to test process.')
            break
        else:
          trigger_times = 0

      self.log("Done!")
      return loss_log


    def __train_model(self):
      self.model.train()
      N = len(self.train_dataloader.dataset)
      steps = N // (self.train_dataloader.batch_size*self.accumulation_steps) 
      step = 0
      true_labels = []
      pred_labels = []

      for index, batch in enumerate(self.train_dataloader):               
        model_input = self.__get_model_input(batch)
        loss, prediction, attention = self.model(**model_input)

        loss /= self.accumulation_steps   
        loss.backward()

        preds = prediction.detach().cpu().numpy()
        true_labels.extend(model_input['labels'].tolist())
        pred_labels.extend(preds.tolist())

        # weights update
        if ((index+1) % self.accumulation_steps == 0) or (step == steps):
          self.optimizer.step()
          self.optimizer.zero_grad()
          self.loss_history.append(loss)
          step += 1
      
      self.__compute_metrics(pred_labels, true_labels, loss, "Train")
        
      return loss


    def __valid_model(self):
      self.log("")
      self.model.eval()
      N = len(self.valid_dataloader.dataset)
      steps = N // self.valid_dataloader.batch_size
      true_labels = []
      pred_labels = []
      with torch.no_grad():
        for index, batch in enumerate(self.valid_dataloader):

          model_input = self.__get_model_input(batch)
          loss, prediction, attention = self.model(**model_input)

          preds = prediction.detach().cpu().numpy()
          true_labels.extend(model_input['labels'].tolist())
          pred_labels.extend(preds.tolist())

      self.__compute_metrics(pred_labels, true_labels, loss, "Validation")

      return loss
    

    def test_model(self, model_save_path, visualize_attention = False
        , visualize_layers= [11], visualize_result_path=""):
      self.log(("-" * 30) + "\n")        
      self.model = torch.load(model_save_path).to(self.device)
      self.model.eval()
      self.log(model_save_path + " load successfully.")
      
      true_labels = []
      pred_labels = []
      with torch.no_grad():
        for index, batch in enumerate(self.test_dataloader):

          model_input = self.__get_model_input(batch)
          loss, prediction, attention = self.model(**model_input)
                   
          preds = prediction.detach().cpu().numpy()
          true_labels.extend(model_input['labels'].tolist())
          pred_labels.extend(preds.tolist())

          if visualize_attention:
            self.__shap_visualize(model_input)
            # for layer in visualize_layers:
            #   self.log("Start visualize token 2 token scores for item " + str(index) + " ...")
            #   all_tokens = self.vectorizer.convert_ids_to_tokens(pair_sequence_features["input_ids"][0])  # Convert input ids to token strings
            #   output_attentions_all = torch.stack(attention)
            #   self.__visualize_token2token_scores(output_attentions_all[layer].squeeze().detach().cpu().numpy()
            #       , all_tokens, visualize_result_path + str(layer) + "_token2token.png")

      result_metrics= self.__compute_metrics(pred_labels, true_labels, loss, "Test")
       
      return loss, result_metrics, pred_labels


    def __get_model_input(self, info):
      pair_sequence_features = {k: v.to(self.device) for k, v in info['pair_sequence_features'].items()}
      claim_features = {k: v.to(self.device) for k, v in info['claim_features'].items()}
      text_features = {k: v.to(self.device) for k, v in info['text_features'].items()}
      
      labels = info['label']
      model_input = {'transformer_input_tensor':pair_sequence_features,
                        'claim_tensor':claim_features, 
                        'text_tensor': text_features,
                        'manual_extracted_features': info['manual_extracted_features'].to(self.device),
                        'labels':labels.to(self.device),
                        'sim_labels':info['similarity_label'].to(self.device),
                        'loss_fn':self.loss_fn}

      return model_input


    def __compute_metrics(self, prediction, ground_truth, loss, title):
      accuracy = metrics.accuracy_score(ground_truth, prediction)
      precision = metrics.precision_score(ground_truth, prediction, zero_division=0
                                          , average='weighted')
      recall = metrics.recall_score(ground_truth, prediction, zero_division=0
                                    , average='weighted')
      f1 = metrics.f1_score(ground_truth, prediction, zero_division=0, average='weighted')
      result_metrics = {'acc' : accuracy, 'pre': precision, 'rec':recall, 'f1': f1,
                        "pre_macro": metrics.precision_score(ground_truth, prediction, zero_division=0, average='macro')
                        , "rec_macro": metrics.recall_score(ground_truth, prediction, zero_division=0, average='macro')
                        , "f1_macro": metrics.f1_score(ground_truth, prediction, zero_division=0, average='macro')}

      self.log(title + " Metrics: \n")
      self.log("loss: " + str(loss.item()))      
      for k, v in result_metrics.items():
        self.log(str(k) + ": " + str(v))
              
      self.log("Confusion matrix:")
      self.log(confusion_matrix(ground_truth, prediction))

      return result_metrics


    def __visualize_token2token_scores(self, scores_mat, all_tokens, 
        visualize_result_path, x_label_name='Head'):
      fig = plt.figure(figsize=(100, 100))

      for idx, scores in enumerate(scores_mat):
        scores_np = np.array(scores)
        ax = fig.add_subplot(4, 3, idx+1)
        # append the attention weights
        im = ax.imshow(scores, cmap='viridis')

        fontdict = {'fontsize': 10}

        ax.set_xticks(range(len(all_tokens)))
        ax.set_yticks(range(len(all_tokens)))

        ax.set_xticklabels(all_tokens, fontdict=fontdict, rotation=90)
        ax.set_yticklabels(all_tokens, fontdict=fontdict)
        ax.set_xlabel('{} {}'.format(x_label_name, idx+1))

        fig.colorbar(im, fraction=0.046, pad=0.04)
      plt.tight_layout()
      plt.show()
      plt.savefig(visualize_result_path)


    def __visualize_token2head_scores(self, scores_mat, all_tokens, visualize_result_path):
      fig = plt.figure(figsize=(50, 50))

      for idx, scores in enumerate(scores_mat):
        scores_np = np.array(scores)
        ax = fig.add_subplot(6, 2, idx+1)
        # append the attention weights
        im = ax.matshow(scores_np, cmap='viridis')

        fontdict = {'fontsize': 20}

        ax.set_xticks(range(len(all_tokens)))
        ax.set_yticks(range(len(scores)))

        ax.set_xticklabels(all_tokens, fontdict=fontdict, rotation=90)
        ax.set_yticklabels(range(len(scores[0])), fontdict=fontdict)
        ax.set_xlabel('Layer {}'.format(idx+1))

        fig.colorbar(im, fraction=0.046, pad=0.04)
      plt.tight_layout()
      plt.show() 
      plt.savefig(visualize_result_path)     


    def __shap_visualize(self, model_input):
      attrib_data = self.__get_model_input(self.train_dataloader[0])
      explainer = shap.DeepExplainer(self.model, **attrib_data)
      shap_vals = explainer.shap_values(**model_input)
      all_tokens = self.vectorizer.convert_ids_to_tokens(model_input['pair_sequence_features']["input_ids"][0])  # Convert input ids to token strings
      shap.summary_plot(shap_vals, feature_names=all_tokens)


    def fit_text_classification(self, batch_size, train_dataset, val_dataset
        , num_epochs, model_save_path, test_dataset):      
      
      training_args = TrainingArguments(
              output_dir=model_save_path,
              overwrite_output_dir=True,
              do_train=True,
              do_eval=True,
              do_predict=True,
              per_device_train_batch_size=batch_size,
              per_device_eval_batch_size =batch_size,
              gradient_accumulation_steps=self.accumulation_steps, 
              learning_rate=3e-5,
              num_train_epochs=num_epochs,
              evaluation_strategy='epoch',
              logging_steps=8,
              eval_steps=1000,
              save_steps=1000,
              load_best_model_at_end= True,
              weight_decay= 0.8e-5,
              save_strategy= "epoch",
              metric_for_best_model = "f1-micro",
              greater_is_better= True,
              save_total_limit = 8,
              # optim = self.optimizer
      )
      trainer = CustomTrainer(
          model=self.model,
          args=training_args,
          train_dataset= train_dataset,
          eval_dataset= val_dataset,
          compute_metrics=self.__compute_metrics, 
      )
      trainer.loss_fn = self.loss_fn
      trainer.train()
      
      trainer.save_model(model_save_path)

      predictions = trainer.predict(test_dataset)
      preds = np.argmax(predictions.predictions, axis=-1)
      self.log(preds)
      self.log(confusion_matrix(test_dataset._target_lebel, preds))
      
      wrongs = []
      for idx in range(len(test_dataset._target_lebel)):
          if test_dataset._target_lebel[idx] != preds[idx]:
              wrongs.append(idx)
      
      acc = metrics.accuracy_score(test_dataset._target_lebel, preds)
      f1 = metrics.f1_score(test_dataset._target_lebel, preds, average='micro')
      f1_macro = metrics.f1_score(test_dataset._target_lebel, preds, average='macro')
      f1_all = metrics.f1_score(test_dataset._target_lebel, preds, average=None)
      self.log({'accuracy': acc,'f1-micro': f1,'f1-macro': f1_macro})
      self.log(f1_all)
      print(wrongs)


class CustomTrainer(Trainer):
  # def __init__(self, loss_fn):
  #   self.loss_fn = loss_fn
  #   super(CustomTrainer, self).__init__()


  def compute_loss(self, model, inputs, return_outputs=False):
    labels = inputs.get("labels")
    # forward pass
    outputs = model(**inputs)
    logits = outputs.get("logits")
    loss = self.loss_fn(logits.view(-1, self.model.config.num_labels), labels.view(-1))
    return (loss, outputs) if return_outputs else loss      