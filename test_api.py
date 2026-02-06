

import requests

response = requests.get(
  'https://api.stormglass.io/v2/weather/point',
  params={
    'lat': 58.7984,
    'lng': 17.8081,
    'params': 'windSpeed',
  },
  headers={
    'Authorization': 'bf6dd17a-0336-11f1-b866-0242ac120004-bf6dd224-0336-11f1-b866-0242ac120004'
  }
)

# Do something with response data.
json_data = response.json()
print(json_data)