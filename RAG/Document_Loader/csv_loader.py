from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='demo_csv.csv')

docs = loader.load()

print(len(docs))
print(docs[10])

