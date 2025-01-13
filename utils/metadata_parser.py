import base64
import json

def parse_image_metadata(img):
    try:
        # Access metadata stored in the image
        metadata = img.info

        # Extract the base64-encoded 'chara' field
        chara_encoded = metadata.get("chara", "")
        if not chara_encoded:
            print("No 'chara' field found in metadata.")
            return "", "", ""

        # Decode the base64 string
        chara_decoded = base64.b64decode(chara_encoded).decode('utf-8')

        # Parse the decoded string as JSON
        chara_data = json.loads(chara_decoded)

        # Extract the required fields
        name = chara_data.get("name", "")
        context = chara_data.get("description", "")  # Use 'description' as 'context'
        greeting = chara_data.get("first_mes", "")  # Use 'first_mes' as 'greeting'

        print(f"Name: {name}")
        print(f"Context: {context}")
        print(f"Greeting: {greeting}")

        # Return the parsed metadata as separate values
        return name, context, greeting
    except Exception as e:
        print(f"Error parsing image metadata: {e}")
        return "", "", ""  # Return empty values if there's an error