import os

api_key = os.getenv("TOGETHER_API_KEY")
print(f'api-key: {api_key}')


str_arr = ["a", "b", "c", "d", "e", "f"]
str = ""
for i in str_arr:
    str += ", " + i
print(str[2:])

print(", ".join(str_arr))

print(str_arr)
print(str_arr[1:])
print(str_arr)
print(str_arr[:1] + str_arr[2:])
print(str_arr)
