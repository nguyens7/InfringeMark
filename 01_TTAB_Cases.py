import xml.etree.ElementTree as ET
import pandas as pd


root = ET.parse('Data/USPTO_TTAB_decisions.xml.xml').getroot()

tags = {"tags":[]}
for elem in root:
    tag = {}
    tag["Id"] = elem.attrib['Id']
    tag["TagName"] = elem.attrib['TagName']
    tag["Count"] = elem.attrib['Count']
    tags["tags"]. append(tag)

df_users = pd.DataFrame(tags["tags"])
df_users.head()