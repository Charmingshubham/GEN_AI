from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path = 'demo_pdfs',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

docs = loader.load()

print(docs[0].metadata)

#we can use lazy_load here beacuse with lots of document lazy load loads a single object documnet at
# a time and them move to second and it better then load() because it loads whole documnts object at 
# time in the main memory and its slow.