import pandas

dataset = pandas.read_csv('dataset.csv')

# print(dataset)

from dotenv import load_dotenv
load_dotenv()


from openai import OpenAI
client = OpenAI()

def simple_call(prompt):
  completions = client.chat.completions.create(model="gpt-3.5-turbo", 
            messages=[
              {"role": "user", "content": prompt},
            ], max_tokens=200
            , temperature=0.1
            , top_p=1
            )
  return completions.choices[0].message.content
  

dataset['positive'] = dataset['reviewtext'].apply(lambda x: simple_call("In a scale of 0 to 10, how positive is the following review for a programmer: \"" + x + "\", answer in one number."   ))
dataset['emomtional'] = dataset['reviewtext'].apply(lambda x: simple_call("In a scale of 0 to 10, how emomtional is the reviewer who wrote the following review: \"" + x + "\", answer in one number."   ))
dataset['sarcastic'] = dataset['reviewtext'].apply(lambda x: simple_call("In a scale of 0 to 10, how sarcastic is the following review: \"" + x + "\", answer in one number."   ))


dataset.to_csv('dataset_processed.csv')










# result = simple_call("What's the capital of the United States? Shortest possible answer.")
# print(result)



                      