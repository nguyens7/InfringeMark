import xml.etree.ElementTree as ET
import pandas as pd


# tree = ET.parse('Documents/com~apple~CloudDocs/InfringeMark/Data.nosyc/USPTO_TTAB_decisions.xml')

# root = tree.getroot()

# print(root)

# tags = {"tags":[]}
# for elem in root:
#     tag = {}
#     tag["Id"] = elem.attrib['Id']
#     tag["TagName"] = elem.attrib['TagName']
#     tag["Count"] = elem.attrib['Count']
#     tags["tags"]. append(tag)

# df_users = pd.DataFrame(tags["tags"])
# df_users.head()

import traceback
# ...
try:
    input_fname = "/Data.nosyc/USPTO_TTAB_decisions.xml"
    tree.parse(input_fname)
    # ...
except IOError:
    ex_info = traceback.format_exc()
    print('ERROR!!! Cannot parse file: %s' % (input_fname))
    print('ERROR!!! Check if this file exists and you have right to read it!')
    print('ERROR!!! Exception info:\n%s' % (ex_info))