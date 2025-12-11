import logging
from typing import Union, Dict, List
import xml.etree.ElementTree as ET
from html import escape

from bioblend.galaxy.objects.wrappers import Workflow, History, Tool

from app.GX_integration.data_manager import DataManager


class ToolFormGenerator:
    """Parses a Galaxy tool's XML definition to generate an HTML form."""
    
    def __init__(self, xml_string: str, data_manager: DataManager, tool: Tool, history: History):
        self.root = ET.fromstring(xml_string)
        self.tool = tool
        self.history = history
        self.data_manger = data_manager
        self.datasets, self.dataset_collections = self.data_manger.list_contents(self.history)
        self.tool_name = self.tool.name
        self.script_blocks = set()
        self.log = logging.getLogger(__class__.__name__)
        self.upload_form_counter = 0 
        self.has_upload_forms = False  # To flag to track if we have upload forms
        
    def build_html(self) -> str:
        """Generates the full HTML form."""

        self.log.info(f"building html form for tool: {self.tool_name}")
        
        try:
            form_body = self._traverse(self.root.find('inputs'))
            script_content = "\n".join(self.script_blocks)
            
            # Add upload toggle script if we have upload forms
            if self.has_upload_forms:
                script_content += self._add_upload_toggle_script()
            
            return f"""
                    <form id="tool-form" class="galaxy-form" action="/api/tools/{self.tool.id}/histories/{self.history.id}/execute" method="POST">
                        {form_body}
                        <button class="galaxy-submit" type="submit">Run Tool</button>
                    </form>

                    <script>
                        {script_content}
                    </script>
                    """
        except Exception as e:
            self.log.error(f"Exception occurred: {e}")
            raise

    def _traverse(self, element) -> str:
        """Recursively traverses the XML tree and builds HTML."""

        html_parts = []
        for child in element:
            if child.tag == 'param':
                html_parts.append(self._build_param(child))
            elif child.tag == 'section':
                html_parts.append(self._build_section(child))
            elif child.tag == 'conditional':
                html_parts.append(self._build_conditional(child))
            # Repeats are more complex and best handled with a frontend framework
            # but this provides a starting point.
        return "\n".join(html_parts)
        
    def _build_param(self, param) -> str:
        """Builds HTML for a single <param> element."""

        param_type = param.get('type')
        name = param.get('name')
        arg_key = param.get('argument') or name
        label = param.get('label', name)
        help_text = param.find('help')
        help_html = f'<small class="form-help-text">{escape(help_text.text.strip())}</small>' if help_text is not None and help_text.text else ''
        input_html = ''
        
        if param_type == 'data':
            # Generate select options for existing datasets
            options = []
            for ds in self.datasets:
                label = f"{ds['hid']}: {ds['name']}"
                option = f'<option value="{ds["id"]}">{escape(label)}</option>'
                options.append(option)
            options = "".join(options)
            
            # Create unique ID for this upload form
            self.upload_form_counter += 1
            unique_id = f"upload-file-form-{self.upload_form_counter}"
            self.has_upload_forms = True
            
            # Create upload button and hidden form
            upload_button = f'<button type="button" class="galaxy-upload-btn" data-target="{unique_id}">Upload File</button>'
            upload_form = f"""
            <div id="{unique_id}" class="upload-form" style="display:none;">
                <form action="/api/histories/{self.history.id}/upload-file" method="POST" enctype="multipart/form-data">
                    <input type="file" name="file" required>
                    <button type="submit">Upload</button>
                </form>
            </div>
            """
            
            input_html = f'<select name="{arg_key}" class="form-control">{options}</select>{upload_button}{upload_form}'
            
        elif param_type == "data_collection":
            # Generate select options for existing collections
            options = []
            for dsc in self.dataset_collections:
                label = f"{dsc['hid']}: {dsc['name']}"
                option = f'<option value="{dsc["id"]}">{escape(label)}</option>'
                options.append(option)
            options = "".join(options)
            # Create unique ID for this upload form
            self.upload_form_counter += 1
            unique_id = f"upload-collection-form-{self.upload_form_counter}"
            self.has_upload_forms = True
            
            # Create upload button and hidden form
            upload_button = f'<button type="button" class="galaxy-upload-btn" data-target="{unique_id}">Upload Collection</button>'
            upload_form = f"""
            <div id="{unique_id}" class="upload-form" style="display:none;">
                <form action="/api/histories/{self.history.id}/upload-collection" method="POST" enctype="multipart/form-data">
                    <input type="file" name="files" multiple required>
                    <select name="collection_type" required>
                        <option value="list">List</option>
                        <option value="paired">Paired</option>
                        <option value="list:paired">List:Paired</option>
                    </select>
                    <input type="text" name="collection_name" placeholder="Collection name (optional)">
                    <textarea name="structure" placeholder="For list:paired, specify structure as JSON (e.g., [[\"file1.fastq\", \"file2.fastq\"]])"></textarea>
                    <button type="submit">Upload Collection</button>
                </form>
            </div>
            """
            
            input_html = f'<select name="{arg_key}" class="form-control">{options}</select>{upload_button}{upload_form}'
            
        elif param_type == 'text':
            value = param.get('value', '')
            input_html = f'<input type="text" name="{arg_key}" value="{escape(value)}" class="form-control">'
            
        elif param_type in ('integer', 'float'):
            value = param.get('value', '')
            input_html = f'<input type="number" name="{arg_key}" value="{escape(value)}" class="form-control" {"step=any" if param_type == "float" else ""}>'
            
        elif param_type == 'boolean':
            checked = param.get('checked', 'false').lower() == 'true'
            input_html = f"""
                        <input type="hidden" name="{arg_key}" value="false">
                        <input type="checkbox" name="{arg_key}" value="true" {'checked' if checked else ''} class="form-check-input">
                        """
            
        elif param_type == 'select':
            param_options = param.find(".//options")
            if param_options is not None:
                data_table_name = param_options.get('from_data_table')
                if data_table_name:
                    options = "".join(
                            f'<option value="{escape(row["value"])}">{escape(row["name"])}</option>'
                            for _, row in self.data_manger.get_data_table_elements(data_table_name).iterrows()
                        )
                else:
                    # Handle static options within <options>
                    options = "".join([
                        f'<option value="{escape(opt.get("value"))}" {"selected" if opt.get("selected") else ""}>{escape(opt.text)}</option>'
                        for opt in param_options.findall('option')
                    ])
            else:
                # Generate from direct <option> children
                options = "".join([
                    f'<option value="{escape(opt.get("value"))}" {"selected" if opt.get("selected") else ""}>{escape(opt.text)}</option>'
                    for opt in param.findall('option')
                ])
            input_html = f'<select name="{arg_key}" class="form-control">{options}</select>'
            
        else:
            return f''
            
        return f"""
                <div class="form-field">
                    <label for="{arg_key}" class="form-label">{escape(label)}</label>
                    {input_html}
                    {help_html}
                </div>
                 """
                
    def _build_section(self, section) -> str:
        """Builds HTML for a <section> element."""

        title = section.get('title', 'Section')
        body = self._traverse(section)
        return f"""
                <fieldset class="form-section">
                    <legend class="w-auto px-2">{escape(title)}</legend>
                    {body}
                </fieldset>
                """
                
    def _build_conditional(self, cond) -> str:
        """Builds HTML and JS for a <conditional> element."""

        cond_name = cond.get('name')
        test_param_xml = cond.find('param')
        test_param_html = self._build_param(test_param_xml)
        test_param_name = test_param_xml.get('name')
        cases_html = []
        for i, when in enumerate(cond.findall('when')):
            value = when.get('value')
            case_body = self._traverse(when)
            # Wrapper div for each case, hidden by default
            cases_html.append(f'<div id="case-{cond_name}-{value}" class="conditional-case" style="display: none;">{case_body}</div>')
        # Add JavaScript to control visibility
        self._add_conditional_script(cond_name, test_param_name)
        return f"""
                <fieldset class="form-section">
                    <legend class="w-auto px-2">Conditional: {escape(cond_name)}</legend>
                    {test_param_html}
                    {''.join(cases_html)}
                </fieldset>
                """
         
    def _add_upload_toggle_script(self):
        """Returns the JavaScript for toggling upload forms."""

        return """
        document.addEventListener('DOMContentLoaded', function() {
            document.querySelectorAll('.galaxy-upload-btn').forEach(button => {
                button.addEventListener('click', function() {
                    const targetId = this.getAttribute('data-target');
                    const form = document.getElementById(targetId);
                    if (form) {
                        form.style.display = form.style.display === 'none' ? 'block' : 'none';
                    }
                });
            });
        });
        """
    
    def _add_conditional_script(self, cond_name: str, test_param_name: str):
        """Adds a JavaScript block for handling conditional logic."""

        script = f"""
                function handleConditional_{cond_name}() {{
                    const selector = document.querySelector('[name="{test_param_name}"]');
                    const selectedValue = selector.value;
                    document.querySelectorAll('#tool-form .conditional-case[id^="case-{cond_name}-"]').forEach(el => {{
                        el.style.display = 'none';
                    }});
                    const activeCase = document.getElementById(`case-{cond_name}-${{selectedValue}}`);
                    if (activeCase) {{
                        activeCase.style.display = 'block';
                    }}
                }}
                document.addEventListener('DOMContentLoaded', function() {{
                    const selector = document.querySelector('[name="{test_param_name}"]');
                    selector.addEventListener('change', handleConditional_{cond_name});
                    // Initial call to set state
                    handleConditional_{cond_name}();
                }});
                """
        self.script_blocks.add(script)

class WorkflowFormGenerator:
    """Workflow html form generator for a galaxy workflow"""

    def __init__ (self, mapped_workflow: Union[List, Dict], data_manager: DataManager, history: History, workflow: Workflow):
        self.mapped_workflow = mapped_workflow
        self.workflow = workflow
        self.history = history
        self.data_manager = data_manager
        self.upload_form_counter = 0
        self.has_upload_forms = False
        self.script_blocks = set()
        
    def _build_html(self) -> str:
        """
        Builds a semantic HTML form from a list of mapped workflow inputs.

        This version improves on the original by:
        - Removing non-standard HTML (<br>, custom attributes).
        - Adding support for integer and float parameter types.
        - Refactoring logic into smaller helper methods for clarity.
        - Correctly handling select lists for all parameter types, which implicitly
          handles cases like reference genomes when they are defined as a select parameter.

        Args:
            mapped_workflows (Union[list, dict]): A single or list of mapped workflow
                                                  input dictionaries.
            history (History): The history to source datasets and collections from.

        Returns:
            str: A complete HTML form as a string.
        """
        if isinstance(self.mapped_workflow, dict):
            self.mapped_workflow = [self.mapped_workflow]

        # Fetch datasets and collections once for performance.
        datasets, collections = self.data_manager.list_contents(history=self.history)
        history_id = self.history.id
        workflow_id = self.workflow['id']
        
        form_parts = [f'<form class="galaxy-form" action="/api/workflows/{workflow_id}/histories/{history_id}/execute" method="POST">']

        for workflow_idx, workflow_inputs in enumerate(self.mapped_workflow):
            form_parts.append(f'<fieldset class="form-section" id="workflow-{workflow_idx}"><legend>{workflow_inputs["name"]}</legend>')

            for _, input_info in workflow_inputs.items():
                if not isinstance(input_info, dict):  # Skip the 'name' key
                    continue
                
                # Generate a unique field name for the input element
                field_name = f"{input_info['step_id']}"
                input_type = input_info.get("input_type", "")
                
                # Create the form field HTML based on the input type
                field_html = ""
                if input_type == "data_input":
                    field_html = self._create_data_input(field_name, datasets)
                elif input_type == "data_collection_input":
                    field_html = self._create_collection_input(field_name, collections)
                elif input_type == "parameter_input":
                    field_html = self._create_parameter_input(field_name, input_info.get("tool_inputs", {}))
                else:   
                    field_html = f'<input type="text" class="form-control" name="{field_name}" placeholder="Unsupported input type: {input_type}">'

                # Assemble the full field with its label and annotation
                form_parts.append(self._assemble_form_field(field_name, input_info, field_html))

            form_parts.append("</fieldset>")

        form_parts.append('<button type="submit" class="galaxy-submit">Run Workflow</button>')
        form_parts.append("</form>")
        
        # Add script blocks if any
        script_content = "\n".join(self.script_blocks)
        if self.has_upload_forms:
            script_content += self._add_upload_toggle_script()
        
        joined_parts = "\n".join(form_parts)
        return  f"""
                {joined_parts}
                <script>
                    {script_content}
                </script>
                """

    def _assemble_form_field(self, field_name: str, input_info: dict, field_html: str) -> str:
        """Helper to wrap an input element with its label and annotations."""
        label = input_info.get("Label", f"Input {field_name}")
        annotation = input_info.get("annotation", "")
        annotation_html = f'<small class="form-help-text">{annotation}</small>' if annotation else ''
        
        return f"""
            <div class="form-field" id="field-wrapper-{field_name}">
                <label for="{field_name}" class="form-label">{label}</label>
                {annotation_html}
                {field_html}
            </div>
            """

    def _create_data_input(self, name: str, datasets: list) -> str:
        """Creates a <select> for a single dataset input with upload functionality."""
        # Generate select options for existing datasets
        options = ''.join([f'<option value="{opt["id"]}">{opt["name"]}</option>' for opt in datasets])
        
        # Create unique ID for this upload form
        self.upload_form_counter += 1
        unique_id = f"upload-file-form-{self.upload_form_counter}"
        self.has_upload_forms = True
        
        # Create upload button and hidden form
        upload_button = f'<button type="button" class="galaxy-upload-btn" data-target="{unique_id}">Upload File</button>'
        upload_form = f"""
        <div id="{unique_id}" class="upload-form" style="display:none;">
            <form action="/api/histories/{self.history.id}/upload-file" method="POST" enctype="multipart/form-data">
                <input type="file" name="file" required>
                <button type="submit">Upload</button>
            </form>
        </div>
        """
        
        return f'<select class="form-control" name="{name}" id="{name}" required>{options}</select>{upload_button}{upload_form}'
    def _create_collection_input(self, name: str, collections: list) -> str:
        """Creates a <select> for a dataset collection input with upload functionality."""
        # Generate select options for existing collections
        options = ''.join([f'<option value="{opt["id"]}">{opt["name"]}</option>' for opt in collections])
        
        # Create unique ID for this upload form
        self.upload_form_counter += 1
        unique_id = f"upload-collection-form-{self.upload_form_counter}"
        self.has_upload_forms = True
        
        # Create upload button and hidden form
        upload_button = f'<button type="button" class="galaxy-upload-btn" data-target="{unique_id}">Upload Collection</button>'
        upload_form = f"""
        <div id="{unique_id}" class="upload-form" style="display:none;">
            <form action="/api/histories/{self.history.id}/upload-collection" method="POST" enctype="multipart/form-data">
                <input type="file" name="files" multiple required>
                <select name="collection_type" required>
                    <option value="list">List</option>
                    <option value="paired">Paired</option>
                    <option value="list:paired">List:Paired</option>
                </select>
                <input type="text" name="collection_name" placeholder="Collection name (optional)">
                <textarea name="structure" placeholder="For list:paired, specify structure as JSON (e.g., [[\"file1.fastq\", \"file2.fastq\"]])"></textarea>
                <button type="submit">Upload Collection</button>
            </form>
        </div>
        """
        
        return f'<select class="form-control" name="{name}" id="{name}" required>{options}</select>{upload_button}{upload_form}'
    def _create_parameter_input(self, name: str, tool_inputs: dict) -> str:
        """Creates an appropriate input element for a workflow parameter."""
        param_type = tool_inputs.get("parameter_type", "text")
        required = "" if tool_inputs.get("optional", False) else "required"

        # Case 1: Boolean parameter (True/False)
        if param_type == "boolean":
            return f"""
                    <select class="form-control" name="{name}" id="{name}" {required}>
                        <option value="true">True</option>
                        <option value="false" selected>False</option>
                    </select>
                    """

        # Case 2: Select/dropdown from a list of choices with restrictions.
        if "restrictions" in tool_inputs:
            options = ''.join([f'<option value="{opt}">{opt}</option>' for opt in tool_inputs["restrictions"]])
            return f'<select class="form-control" name="{name}" id="{name}" {required}>{options}</select>'
        
        # Case 3: Number inputs (Integer or Float)
        if param_type == "integer":
            return f'<input type="number" class="form-control" name="{name}" id="{name}" {required}>'
        if param_type == "float":
            return f'<input type="number" step="any" class="form-control" name="{name}" id="{name}" {required}>'
            
        # Case 4: Default to a simple text input
        return f'<input type="text" class="form-control" name="{name}" id="{name}" {required}>'
    
    def _add_upload_toggle_script(self):
        """Returns the JavaScript for toggling upload forms."""
        return """
        document.addEventListener('DOMContentLoaded', function() {
            document.querySelectorAll('.galaxy-upload-btn').forEach(button => {
                button.addEventListener('click', function() {
                    const targetId = this.getAttribute('data-target');
                    const form = document.getElementById(targetId);
                    if (form) {
                        form.style.display = form.style.display === 'none' ? 'block' : 'none';
                    }
                });
            });
        });
        """
    
# TODO: Add also a create collection button and also validate the upload buttons/forms against the forms.