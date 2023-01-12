import turicreate as tc

sf = tc.SFrame("people_wiki.sframe")

print(sf) # shows the hall data
print(sf.num_rows()) # shows the amount of the data rows
print(sf["name"].tail(1)) # shows the last data in the name column
print(sf[(sf['name'] == 'Harpdog Brown')]['text']) # shows the text linked to the Harpdog Brown
print(sf.sort('text'))
print(sf.head(1)[0]['name']) # first name in the raw
