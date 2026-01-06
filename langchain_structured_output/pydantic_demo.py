from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class student(BaseModel):
    name:str
    age: Optional[str] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, default=5,description='cgpa of a student represented in scale of 10')

new_student: student = {'name': 'John Doe', 'age': '21', 'email': 'abc@gmail', 'cgpa': 8.5}
print(new_student)

dict_student = dict(new_student)
print(dict_student['age'])