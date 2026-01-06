import pkgutil
import langchain_community.retrievers as r

print([p.name for p in pkgutil.iter_modules(r.__path__)])
