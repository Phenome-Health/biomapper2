import json

from biomapper2.mapper import Mapper


mapper = Mapper()

# Define an entity with a name and local identifiers
item = {"name": "carnitine", "kegg": ["C00487"], "pubchem": "10917"}

# Print out the original entity
print(f"\nOriginal entity:")
print(json.dumps(item, indent=2))

# Run through the full KG mapping pipeline
mapped_item = mapper.map_entity_to_kg(
    item=item, name_field="name", provided_id_fields=["kegg", "pubchem"], entity_type="metabolite"
)

# Print the final mapped entity
print(f"\nMapped entity:")
print(json.dumps(mapped_item, indent=2))
