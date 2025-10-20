curl --request GET \
  --url https://api.venice.ai/api/v1/image/styles

Image Styles
List available image styles that can be used with the generate API.
​
Authorization
stringheaderrequired
Bearer authentication header of the form Bearer <token>, where <token> is your auth token.

Response

200

application/json
OK

​
data
string[]required
List of available image styles

Example:
[
  "3D Model",
  "Analog Film",
  "Anime",
  "Cinematic",
  "Comic Book"
]
​
object
enum<string>required
Available options: list 