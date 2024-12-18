from dadmatools.normalizer import Normalizer
import re


class Preprocessor():

  def __init__(self, remove_stop_word = False, remove_puncs= False):
    self.normalizer = Normalizer(
          full_cleaning=False,
          unify_chars=True,
          refine_punc_spacing=True,
          remove_extra_space=True,
          remove_puncs= remove_puncs,
          remove_html=True,
          remove_stop_word= remove_stop_word,
          replace_email_with="<EMAIL>",
          replace_number_with=None,
          replace_url_with=None,
          replace_mobile_number_with=None,
          replace_emoji_with="",
          replace_home_number_with=None
      )
    
    shayee = self.normalizer.normalize("شایعه")
    re_pattern1 = "(/(\s)*"+ shayee +"(\s)*[0-9]+)|(/(\s)*شایعه(\s)*[0-9]+)"
    # re_pattern2 = "/(\s)*[0-9]+"
    re_pattern3 = "\\u200c|\\u200d|\\u200e|\\u200b|\\u2067|\\u2069"

    self.digit_convertor = {
    '۰' : '0',
    '۱' : '1',
    '۲' : '2',
    '۳' : '3',
    '۴' : '4',
    '۵' : '5',
    '۶' : '6',
    '۷' : '7',
    '۸' : '8',
    '۹' : '9'
    }

    self.emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
        "]+", flags=re.UNICODE)

    self.pattern_list = [re_pattern1, re_pattern3]


  def clean_text(self, text):
    
    clean_sentence = self.remove_hashtags(text)

    clean_sentence = self.normalizer.normalize(text)

    for pattern in self.pattern_list:
      clean_sentence = re.sub(pattern, " ", clean_sentence)

    clean_sentence = self.convert_fa_digit_to_en(clean_sentence)

    return clean_sentence
  

  def convert_fa_digit_to_en(self, text):
    
    transTable = text.maketrans(self.digit_convertor)
    text = text.translate(transTable)
    return text


  def remove_emoji(self, text):
    return self.emoji_pattern.sub(r'', text)    


  def remove_hashtags(self, text):
    news_text = [word.strip("#").replace('_', ' ') if word.startswith("#") else word for word in text.split()]
    news_text = ' '.join(news_text)

    return news_text