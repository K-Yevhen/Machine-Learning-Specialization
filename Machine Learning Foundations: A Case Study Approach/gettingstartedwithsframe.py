import turicreate as tc

sf = tc.SFrame('people-example.csv')

print(sf)
print(sf.head())
print(sf.tail())

# Take any data structure in GraphlabCreate
sf.show()
sf.explore()
sf["age"].show()

# inspect columns of dataset
print(sf["Country"])
print(sf["age"])
print(sf['age'].mean())
print(sf["age"].max())
sf['Full Name'] = sf['First Name'] + ' ' + sf['Last Name']
print(sf["age"] + 2)
print(sf["age"] * sf["age"])
print(sf)


# input("Press Enter to exit...") it doesn't show and stuck on the loading page
input("Press Enter to exit...")

# Use the apply function to do a advance transformation of our data
print(sf["Country"])
sf["Country"].show()

def transform_country(country):
    if country == "USA":
        return "United States"
    else:
        return country

print(transform_country("Brasil"))

print(sf['Country'].apply(transform_country))

sf['Country'] = sf['Country'].apply(transform_country)

print(sf)
