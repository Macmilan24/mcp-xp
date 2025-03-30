import xml.etree.ElementTree as ET

def extract_inputs_outputs(xml_string):
    """
    Extracts the content of <inputs> and <outputs> tags from an XML string.
    Args:
        xml_string (str): A string containing the XML data.
    Returns:
        tuple: A tuple containing two elements:
            - inputs_content (str or None): The content of the <inputs> tag as a string, or None if the tag is not found.
            - outputs_content (str or None): The content of the <outputs> tag as a string, or None if the tag is not found.
    Raises:
        xml.etree.ElementTree.ParseError: If the XML string cannot be parsed.
    """
    try:
        # Parse the XML string
        root = ET.fromstring(xml_string)

        # Extract content from <inputs> tag
        inputs = root.find('inputs')
        inputs_content = ET.tostring(inputs, encoding='unicode', method='xml').strip() if inputs is not None else None

        # Extract content from <outputs> tag (if it exists)
        outputs = root.find('outputs')
        outputs_content = ET.tostring(outputs, encoding='unicode', method='xml').strip() if outputs is not None else None

        return inputs_content, outputs_content
    except ET.ParseError as e:
        return f"Error parsing XML: {e}", None
