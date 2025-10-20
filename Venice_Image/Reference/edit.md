curl --request POST \
  --url https://api.venice.ai/api/v1/image/edit \
  --header 'Authorization: Bearer <token>' \
  --header 'Content-Type: application/json' \
  --data '{
  "prompt": "Colorize",
  "image": "iVBORw0KGgoAAAANSUhEUgAAAgAAAAIACAIAAAB7GkOtAAAAIGNIUk0A..."
}'

Edit (aka Inpaint)
Edit or modify an image based on the supplied prompt. The image can be provided either as a multipart form-data file upload or as a base64-encoded string in a JSON request.

Authorizations
​
Authorization
stringheaderrequired
Bearer authentication header of the form Bearer <token>, where <token> is your auth token.

Body

application/json
Edit an image based on the supplied prompt.

​
prompt
stringrequired
The text directions to edit or modify the image. Does best with short but descriptive prompts. IE: "Change the color of", "remove the object", "change the sky to a sunrise", etc.

Maximum length: 1500
Example:
"Change the color of the sky to a sunrise"

​
image

any
required
The image to edit. Can be either a file upload, a base64-encoded string, or a URL starting with http:// or https://. Image dimensions must be at least 65536 pixels and must not exceed 33177600 pixels. Image URLs must be less than 10MB.

Response

200

image/png
OK

The response is of type file.