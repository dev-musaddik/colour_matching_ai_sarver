import os

# Name of the main folder
main_folder = "Hair_Colors"
os.makedirs(main_folder, exist_ok=True)

# Your list of colors
colors = [
    "#1 Jet Black",
    "#1B Off-Black",
    "#1C Darkest Brown",
    "#2B Dark Chocolate Brown",
    "#2C Medium Chestnut Brown",
    "#3A Light Warm Brown",
    "#4B Warm Honey Brown",
    "#4C Rich Chestnut",
    "#5A Medium Ash Brown",
    "#8A Light Ash Brown",
    "#10A Medium Golden Blonde",
    "#13A Pale Ash Blonde",
    "#27 Strawberry Blonde",
    "#30A Deep Auburn Copper",
    "#50 Pale Icy Silver",
    "#60A Pure White Platinum",
    "#62 Icy Platinum",
    "#64 Creamy Pearl",
    "#13A/24 Champagne Blend",
    "#2CT5 Mocha Root Melt",
    "#2BT6 Dark Chocolate Dip",
    "#4B/27 Golden Brown Swirl",
    "#4C/27 Cool Espresso Streak",
    "#6/27 Caramel Toast",
    "#613L/18A Vanilla Ash Swirl",
    "#2BT8A Ashy Rooted Bronde"
]

for color in colors:
    # Replace forbidden characters for folder names
    safe_name = color.translate(str.maketrans({
        "#":"_",
        "/":"_",
        "\\":"_",
        ":":"_",
        "*":"_",
        "?":"_",
        "\"":"_",
        "<":"_",
        ">":"_",
        "|":"_"
    }))
    
    folder_path = os.path.join(main_folder, safe_name)
    os.makedirs(folder_path, exist_ok=True)

print(f"All folders created inside '{main_folder}'")
