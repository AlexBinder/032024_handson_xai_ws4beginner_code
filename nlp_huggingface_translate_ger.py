from transformers import pipeline

#helsinki needs sentencepiece, recommended: sacremoses
def run():
  
  ger_en_translator = pipeline("translation_de_to_en", model = 'Helsinki-NLP/opus-mt-de-en')
  
  inp = 'Unter den Gräsern nähre ich mich von Dunkelheit viele Tage schon. Besessen und furchtlos. Kleine Welt, die ich ertasten kann.'
  
  output  =  ger_en_translator(inp)
  
  print(output)

def run1b():

  inp = 'Unter den Gräsern nähre ich mich von Dunkelheit viele Tage schon. Besessen und furchtlos. Kleine Welt, die ich ertasten kann.'

  from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

  tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
  model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-de-en")

  input_ids = tokenizer(inp, return_tensors="pt", add_special_tokens=True)
  
  output_ids = model.generate(input_ids.input_ids, max_new_tokens=200)
  
  print(len(output_ids), output_ids[0] )
  
  output= tokenizer.decode(output_ids[0], skip_special_tokens=True)

  print(  len( output_ids[0]), len( output.split(' ') ) )
 
  
  print(output)

def run2():

 
  from transformers import FSMTForConditionalGeneration, FSMTTokenizer
  mname = "allenai/wmt19-de-en-6-6-base" # there is also a allenai/wmt19-de-en-6-6-big
  tokenizer = FSMTTokenizer.from_pretrained(mname)
  model = FSMTForConditionalGeneration.from_pretrained(mname)

  inp = 'Unter den Gräsern nähre ich mich von Dunkelheit viele Tage schon. Besessen und furchtlos. Kleine Welt, die ich ertasten kann.'
  
  input_ids = tokenizer.encode(inp, return_tensors="pt")
  outputs = model.generate(input_ids)
  decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
  print(decoded)

def run3():

  from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

  tokenizer = AutoTokenizer.from_pretrained("google/bert2bert_L-24_wmt_de_en", pad_token="<pad>", eos_token="</s>", bos_token="<s>")
  model = AutoModelForSeq2SeqLM.from_pretrained("google/bert2bert_L-24_wmt_de_en")

  sentence = "Unter den Gräsern nähre ich mich von Dunkelheit viele Tage schon. Besessen und furchtlos. Kleine Welt, die ich ertasten kann."

  input_ids = tokenizer(sentence, return_tensors="pt", add_special_tokens=False).input_ids
  
  output_ids = model.generate(input_ids)[0]
  res= tokenizer.decode(output_ids, skip_special_tokens=True)

  print(res)

if __name__ == '__main__':
  #run()
  run1b() 
  #run2()
  
  ###### a huge model!
  #run3()
  ##### 
  
