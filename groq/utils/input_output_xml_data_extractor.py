import xml.etree.ElementTree as ET


def extract_input_details(xml_string):
    """
    Extracts details about input data from an XML string.

    Args:
        xml_string: The XML string to parse.

    Returns:
        A list of dictionaries, each representing an input parameter,
        or an error message if parsing fails.
    """
    try:
        root = ET.fromstring(xml_string)
        params = []  # List to store parameter objects

        for param in root.findall('.//param'):
            name = param.get('name')
            data_type = param.get('type')
            format = param.get('format')
            label = param.get('label')

            if name:
                params.append({
                    'name': name,  # Added name to the dictionary
                    'type': data_type,
                    'format': format,
                    'label': label,
                    # Add other relevant attributes as needed
                })

        return params  # Return the list of parameter objects

    except ET.ParseError as e:
        return f"Error parsing XML: {e}"


def extract_output_details(xml_string):
    """
    Extracts details about output data from an XML string.

    Args:
        xml_string: The XML string to parse.

    Returns:
        A list of dictionaries, each representing an output data element,
        or an error message if parsing fails.
    """
    try:
        root = ET.fromstring(xml_string)
        params = []  # List to store data objects

        for data in root.findall('.//data'):
            name = data.get('name')
            data_type = data.get('type')
            format = data.get('format')
            label = data.get('label')

            if name:
                params.append({
                    'name': name,  # Added name to the dictionary
                    'type': data_type,
                    'format': format,
                    'label': label,
                })
        return params  # Return the list of data objects

    except ET.ParseError as e:
        return f"Error parsing XML: {e}"