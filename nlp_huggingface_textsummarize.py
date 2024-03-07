from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def run():

  #text= "The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country. It'll lower the deficit and ask the ultra-wealthy and corporations to pay their fair share. And no one making under $400,000 per year will pay a penny more in taxes."
  
  #text= "A 17-year-old boy has been arrested. A 14-year-old arrested earlier was released without charge. A spokesperson for the fire service said it was alerted to the blaze at Dawn Paper's premises on Donore Road at around 22:00 local time on Saturday. The fire service said it expected to have crews at the site for the rest of the day. The roof of the building has collapsed and what remains of it is expected to smoulder for a number of days. At one point during the night 12 units of the fire brigade tackled the blaze. There are no reports of any casualties. Dawn Paper was established in 1988 and the company manufactures industrial cleaning paper and domestic tissue paper products for the Irish market."
  
  text="Students complained that they might have to code without AI assistance due to privacy concerns of the employing company. A spokesperson for xyz.ai said that their data does not permit to leak content into the hands of owners of large language models. For that reason they require their employees to code on their own." 
  
  text = "summarize: " + text #append command what to do for the model
  
  tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_billsum_model")
  model = AutoModelForSeq2SeqLM.from_pretrained("stevhliu/my_awesome_billsum_model")

  inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True)

  #print(tokenizer.tokenize(text))
  print(inputs.input_ids)
  
  outputs = model.generate(inputs.input_ids, max_new_tokens=100, do_sample=False )
  
  print(outputs[0])
  result = tokenizer.decode(outputs[0], skip_special_tokens=True)
  print(result)


  
if __name__=='__main__':
  run()  
