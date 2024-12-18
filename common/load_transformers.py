from transformers import BigBirdModel, AutoModel


"""## Transformers"""
class LoadTransformers:

  @staticmethod
  def bigbird(model_path, num_labels):
    return BigBirdModel.from_pretrained(model_path, hidden_dropout_prob=0.4
      , block_size=32, classifier_dropout= 0.5, attention_probs_dropout_prob=0.3
      , hidden_act="gelu")


  @staticmethod
  def common_transformers(model_path, num_labels):
      return AutoModel.from_pretrained(model_path, output_attentions=True)


  @staticmethod
  def xlm_roberta(model_path, num_labels):
    return LoadTransformers.common_transformers(model_path, num_labels)


  @staticmethod
  def xlm_roberta_freeze_six_layers(model_path, num_labels):
    model = LoadTransformers.common_transformers(model_path, num_labels)
    for name, value in model.named_parameters():
      if 'encoder.layer.0.' in name or 'encoder.layer.1.' in name or 'encoder.layer.2.' in name or 'encoder.layer.3.' in name or 'encoder.layer.4.' in name or 'encoder.layer.5.' in name:
        value.requires_grad = False
    
    return model


  @staticmethod
  def parsbert(model_path, num_labels):
    return LoadTransformers.common_transformers(model_path, num_labels)


  @staticmethod
  def albert_freeze_9_layers(model_path, num_labels):
    model = LoadTransformers.common_transformers(model_path, num_labels)
    for name, value in model.named_parameters():
      if 'encoder.layer.0.' in name or 'encoder.layer.1.' in name or 'encoder.layer.2.' in name or 'encoder.layer.3.' in name or 'encoder.layer.4.' in name or 'encoder.layer.5.' in name or 'encoder.layer.6.' in name or 'encoder.layer.7.' in name or 'encoder.layer.8.' in name or 'encoder.layer.9.' in name:
        value.requires_grad = False
    
    return model