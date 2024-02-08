
# Usage example

df=pd.read_csv('./data/your_data.csv', nrows=30)
article_processor = ArticleProcessor(df)
query = "write an article titled: [your desired subject]?"
prompt = article_processor.build_prompt(query)
article_processor.generate_completion(prompt)