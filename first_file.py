data = ["Simon", "William", "Emil"]

# Find den person med det første forbogstav
min_name = min(data)  # min() sammenligner alfabetisk
data.remove(min_name)  # Fjern personen fra sin nuværende placering
data.insert(0, min_name)  # Tilføj personen forrest

print(data)
