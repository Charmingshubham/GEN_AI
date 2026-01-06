from typing import TypedDict

class person(TypedDict):
    name: str
    age: int
    email: str

new_person: person = {'name': 'Alice', 'age': 30, 'email': 'ldjnd@gmail'}

print(new_person)