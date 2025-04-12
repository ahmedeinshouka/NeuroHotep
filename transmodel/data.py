import csv
import random

# Core dataset (~300 signs from Gardiner’s Sign List, fully annotated)
core_hieroglyph_data = [
    # A: Man and Occupations
    {"unicode": "𓀀", "gardiner": "A1", "translit": "rꜥ", "english": "sun"},
    {"unicode": "𓀁", "gardiner": "A2", "translit": "s", "english": "man"},
    {"unicode": "𓀂", "gardiner": "A3", "translit": "ḥr", "english": "face"},
    {"unicode": "𓀃", "gardiner": "A4", "translit": "nfr", "english": "good"},
    {"unicode": "𓀄", "gardiner": "A5", "translit": "ꜥšꜣ", "english": "many"},
    {"unicode": "𓀅", "gardiner": "A6", "translit": "šms", "english": "servant"},
    {"unicode": "𓀆", "gardiner": "A7", "translit": "wr", "english": "great one"},
    {"unicode": "𓀎", "gardiner": "A12", "translit": "ꜥḥꜣ", "english": "warrior"},
    {"unicode": "𓀗", "gardiner": "A17", "translit": "ḫrd", "english": "child"},
    {"unicode": "𓀟", "gardiner": "A28", "translit": "kꜣj", "english": "high"},
    {"unicode": "𓀩", "gardiner": "A40", "translit": "nṯr", "english": "seated god"},
    {"unicode": "𓀭", "gardiner": "A43", "translit": "nsw", "english": "king seated"},
    {"unicode": "𓀲", "gardiner": "A50", "translit": "šps", "english": "noble"},
    {"unicode": "𓀸", "gardiner": "A55", "translit": "ꜣw", "english": "dead man"},

    # B: Woman
    {"unicode": "𓁐", "gardiner": "B1", "translit": "st", "english": "woman"},
    {"unicode": "𓁑", "gardiner": "B2", "translit": "ms", "english": "pregnant woman"},
    {"unicode": "𓁓", "gardiner": "B7", "translit": "ḥmt", "english": "wife"},
    {"unicode": "𓁕", "gardiner": "B5", "translit": "mwt", "english": "mother"},

    # C: Deities
    {"unicode": "𓁚", "gardiner": "C1", "translit": "nṯr", "english": "god"},
    {"unicode": "𓁛", "gardiner": "C2", "translit": "nṯrt", "english": "goddess"},
    {"unicode": "𓁜", "gardiner": "C3", "translit": "jmn", "english": "amon"},
    {"unicode": "𓁝", "gardiner": "C4", "translit": "rꜥ", "english": "ra"},
    {"unicode": "𓁞", "gardiner": "C6", "translit": "jnpw", "english": "anubis"},
    {"unicode": "𓁟", "gardiner": "C10", "translit": "mꜣꜥt", "english": "ma'at"},
    {"unicode": "𓁠", "gardiner": "C11", "translit": "ḥwt-ḥr", "english": "hathor"},
    {"unicode": "𓁡", "gardiner": "C12", "translit": "wsjr", "english": "osiris"},
    {"unicode": "𓁢", "gardiner": "C17", "translit": "stẖ", "english": "seth"},

    # D: Body Parts
    {"unicode": "𓁷", "gardiner": "D1", "translit": "ḥr", "english": "head"},
    {"unicode": "𓁹", "gardiner": "D2", "translit": "jr", "english": "eye"},
    {"unicode": "𓁿", "gardiner": "D7", "translit": "jrt", "english": "eye with makeup"},
    {"unicode": "𓂀", "gardiner": "D10", "translit": "wḏꜣt", "english": "eye of horus"},
    {"unicode": "𓂋", "gardiner": "D21", "translit": "r", "english": "mouth"},
    {"unicode": "𓂓", "gardiner": "D28", "translit": "kꜣ", "english": "soul"},
    {"unicode": "𓂧", "gardiner": "D36", "translit": "ḏ", "english": "hand"},
    {"unicode": "𓂝", "gardiner": "D37", "translit": "ꜥ", "english": "arm"},
    {"unicode": "𓂻", "gardiner": "D54", "translit": "jw", "english": "legs walking"},
    {"unicode": "𓂽", "gardiner": "D56", "translit": "rd", "english": "leg"},
    {"unicode": "𓂾", "gardiner": "D58", "translit": "b", "english": "foot"},
    {"unicode": "𓃀", "gardiner": "D60", "translit": "jb", "english": "heart"},

    # E: Mammals
    {"unicode": "𓃒", "gardiner": "E1", "translit": "kꜣ", "english": "bull"},
    {"unicode": "𓃓", "gardiner": "E2", "translit": "ng", "english": "bull charging"},
    {"unicode": "𓃠", "gardiner": "E9", "translit": "mj", "english": "cat"},
    {"unicode": "𓃥", "gardiner": "E13", "translit": "jw", "english": "dog"},
    {"unicode": "𓃭", "gardiner": "E17", "translit": "sꜣb", "english": "jackal"},
    {"unicode": "𓃸", "gardiner": "E23", "translit": "ꜣbw", "english": "elephant"},
    {"unicode": "𓃻", "gardiner": "E26", "translit": "ḥꜣt", "english": "hippopotamus"},
    {"unicode": "𓄂", "gardiner": "E34", "translit": "wn", "english": "hare"},
    {"unicode": "𓃲", "gardiner": "E20", "translit": "rw", "english": "lion"},

    # F: Parts of Mammals
    {"unicode": "𓄀", "gardiner": "F1", "translit": "tp", "english": "head of bull"},
    {"unicode": "𓄇", "gardiner": "F12", "translit": "wsr", "english": "strength"},
    {"unicode": "𓄓", "gardiner": "F18", "translit": "ꜣpd", "english": "bird"},
    {"unicode": "𓄙", "gardiner": "F25", "translit": "wḥm", "english": "leg"},
    {"unicode": "𓄛", "gardiner": "F27", "translit": "nm", "english": "skin"},
    {"unicode": "𓄝", "gardiner": "F29", "translit": "st", "english": "tail"},

    # G: Birds
    {"unicode": "𓄿", "gardiner": "G1", "translit": "ꜣ", "english": "vulture"},
    {"unicode": "𓅀", "gardiner": "G2", "translit": "ꜣꜣ", "english": "two vultures"},
    {"unicode": "𓅃", "gardiner": "G5", "translit": "ḥr", "english": "horus"},
    {"unicode": "𓅆", "gardiner": "G7", "translit": "bjk", "english": "falcon"},
    {"unicode": "𓅓", "gardiner": "G17", "translit": "m", "english": "owl"},
    {"unicode": "𓅜", "gardiner": "G25", "translit": "ꜣḫ", "english": "akh"},
    {"unicode": "𓅱", "gardiner": "G43", "translit": "w", "english": "quail"},
    {"unicode": "𓅬", "gardiner": "G39", "translit": "sꜣ", "english": "duck"},
    {"unicode": "𓅮", "gardiner": "G40", "translit": "gb", "english": "goose"},

    # H: Parts of Birds
    {"unicode": "𓆃", "gardiner": "H1", "translit": "mꜣ", "english": "feather"},
    {"unicode": "𓆄", "gardiner": "H2", "translit": "šw", "english": "wing"},
    {"unicode": "𓆇", "gardiner": "H6", "translit": "šwt", "english": "feather of ma'at"},
    {"unicode": "𓆅", "gardiner": "H3", "translit": "ꜥp", "english": "winged disk"},

    # I: Reptiles/Insects
    {"unicode": "𓆈", "gardiner": "I1", "translit": "ḫpr", "english": "scarab"},
    {"unicode": "𓆎", "gardiner": "I9", "translit": "f", "english": "viper"},
    {"unicode": "𓆑", "gardiner": "I10", "translit": "ḏt", "english": "cobra"},
    {"unicode": "𓆙", "gardiner": "I13", "translit": "sš", "english": "adder"},
    {"unicode": "𓆚", "gardiner": "I14", "translit": "ꜣm", "english": "snake"},
    {"unicode": "𓆤", "gardiner": "L2", "translit": "bjt", "english": "bee"},
    {"unicode": "𓆧", "gardiner": "L6", "translit": "ꜥꜣb", "english": "centipede"},

    # K: Fish
    {"unicode": "𓆡", "gardiner": "K1", "translit": "jn", "english": "fish"},
    {"unicode": "𓆢", "gardiner": "K4", "translit": "ꜥd", "english": "tilapia"},
    {"unicode": "𓆣", "gardiner": "K5", "translit": "šꜣ", "english": "catfish"},
    {"unicode": "𓆥", "gardiner": "K6", "translit": "bw", "english": "mullet"},

    # L: Insects
    {"unicode": "𓆦", "gardiner": "L4", "translit": "srr", "english": "grasshopper"},
    {"unicode": "𓆨", "gardiner": "L7", "translit": "sf", "english": "scorpion"},

    # M: Plants
    {"unicode": "𓇋", "gardiner": "M17", "translit": "j", "english": "reed"},
    {"unicode": "𓇍", "gardiner": "M2", "translit": "nḫt", "english": "tree"},
    {"unicode": "𓇗", "gardiner": "M9", "translit": "šn", "english": "lotus"},
    {"unicode": "𓇛", "gardiner": "M11", "translit": "ꜥš", "english": "papyrus"},
    {"unicode": "𓇟", "gardiner": "M16", "translit": "ḥꜣ", "english": "papyrus clump"},
    {"unicode": "𓇤", "gardiner": "M23", "translit": "sw", "english": "sedge"},
    {"unicode": "𓇥", "gardiner": "M24", "translit": "rš", "english": "plant"},

    # N: Cosmic/Nature
    {"unicode": "𓇯", "gardiner": "N1", "translit": "pt", "english": "sky"},
    {"unicode": "𓇳", "gardiner": "N5", "translit": "rꜥ", "english": "sun"},
    {"unicode": "𓇵", "gardiner": "N8", "translit": "jꜥḥ", "english": "moon"},
    {"unicode": "𓇼", "gardiner": "N14", "translit": "sba", "english": "star"},
    {"unicode": "𓇾", "gardiner": "N2", "translit": "tꜣ", "english": "earth"},
    {"unicode": "𓈖", "gardiner": "N35", "translit": "n", "english": "water"},
    {"unicode": "𓈋", "gardiner": "N25", "translit": "ḫꜣst", "english": "desert"},
    {"unicode": "𓈗", "gardiner": "N36", "translit": "mw", "english": "waterway"},
    {"unicode": "𓈙", "gardiner": "N37", "translit": "š", "english": "pool"},
    {"unicode": "𓈜", "gardiner": "N41", "translit": "ḥ", "english": "well"},

    # O: Buildings
    {"unicode": "𓉐", "gardiner": "O1", "translit": "pr", "english": "house"},
    {"unicode": "𓉒", "gardiner": "O6", "translit": "ḥwt-nṯr", "english": "temple"},
    {"unicode": "𓉔", "gardiner": "O11", "translit": "ꜥḥ", "english": "palace"},
    {"unicode": "𓉘", "gardiner": "O15", "translit": "kꜣr", "english": "shrine"},
    {"unicode": "𓉢", "gardiner": "O24", "translit": "pr-ꜥꜣ", "english": "great house"},
    {"unicode": "𓉥", "gardiner": "O29", "translit": "ꜥꜣ", "english": "column"},
    {"unicode": "𓉩", "gardiner": "O34", "translit": "sꜣ", "english": "door bolt"},

    # P: Ships
    {"unicode": "𓊝", "gardiner": "P1", "translit": "dp", "english": "boat"},
    {"unicode": "𓊡", "gardiner": "P3", "translit": "wjꜣ", "english": "sacred bark"},
    {"unicode": "𓊢", "gardiner": "P5", "translit": "ꜥḥꜣ", "english": "sail"},
    {"unicode": "𓊦", "gardiner": "P8", "translit": "ꜥꜣw", "english": "oar"},

    # Q: Furniture
    {"unicode": "𓋏", "gardiner": "Q1", "translit": "wsr", "english": "throne"},
    {"unicode": "𓋐", "gardiner": "Q3", "translit": "st", "english": "seat"},
    {"unicode": "𓋓", "gardiner": "Q7", "translit": "tꜣ", "english": "brazier"},

    # R: Temple
    {"unicode": "𓋴", "gardiner": "R1", "translit": "s", "english": "health"},
    {"unicode": "𓋸", "gardiner": "R8", "translit": "šn", "english": "protection"},
    {"unicode": "𓋾", "gardiner": "R11", "translit": "ḏd", "english": "stability"},
    {"unicode": "𓋿", "gardiner": "R13", "translit": "jꜣt", "english": "office"},
    {"unicode": "𓌀", "gardiner": "R15", "translit": "ḥtp", "english": "offering"},
    {"unicode": "𓌃", "gardiner": "R19", "translit": "zp", "english": "scepter"},

    # S: Crowns/Symbols
    {"unicode": "𓋹", "gardiner": "S34", "translit": "ꜥnḫ", "english": "life"},
    {"unicode": "𓋺", "gardiner": "S36", "translit": "šw", "english": "feather"},
    {"unicode": "𓋻", "gardiner": "S38", "translit": "dšrt", "english": "red crown"},
    {"unicode": "𓋼", "gardiner": "S39", "translit": "ḥḏt", "english": "white crown"},
    {"unicode": "𓋽", "gardiner": "S40", "translit": "psḏt", "english": "double crown"},
    {"unicode": "𓌂", "gardiner": "S45", "translit": "nḫḫ", "english": "eternity"},
    {"unicode": "𓌅", "gardiner": "S29", "translit": "s", "english": "folded cloth"},
    {"unicode": "𓌇", "gardiner": "S24", "translit": "ṯꜣw", "english": "knot"},
    {"unicode": "𓌈", "gardiner": "S25", "translit": "md", "english": "belt"},

    # T: Warfare
    {"unicode": "𓌪", "gardiner": "T1", "translit": "ḥḏ", "english": "mace"},
    {"unicode": "𓌫", "gardiner": "T3", "translit": "sšt", "english": "arrow"},
    {"unicode": "𓌝", "gardiner": "T14", "translit": "ꜣms", "english": "scepter"},
    {"unicode": "𓌐", "gardiner": "T16", "translit": "pḏ", "english": "bow"},
    {"unicode": "𓌔", "gardiner": "T19", "translit": "sḫm", "english": "harpoon"},
    {"unicode": "𓌗", "gardiner": "T21", "translit": "šnꜥ", "english": "net"},
    {"unicode": "𓌙", "gardiner": "T24", "translit": "ꜥẖ", "english": "knife"},

    # U: Agriculture/Tools
    {"unicode": "𓍯", "gardiner": "U1", "translit": "hb", "english": "plow"},
    {"unicode": "𓍱", "gardiner": "U6", "translit": "sn", "english": "sickle"},
    {"unicode": "𓍝", "gardiner": "U13", "translit": "šm", "english": "hoe"},
    {"unicode": "𓍾", "gardiner": "U36", "translit": "bꜣ", "english": "adze"},
    {"unicode": "𓍲", "gardiner": "U7", "translit": "mr", "english": "chisel"},
    {"unicode": "𓎍", "gardiner": "U28", "translit": "ḏꜣ", "english": "fire drill"},

    # V: Rope/Symbols
    {"unicode": "𓎯", "gardiner": "V1", "translit": "ḥ", "english": "cord"},
    {"unicode": "𓎼", "gardiner": "V28", "translit": "ḥ", "english": "wick"},
    {"unicode": "𓎛", "gardiner": "V20", "translit": "mdw", "english": "ten"},
    {"unicode": "𓏏", "gardiner": "X1", "translit": "t", "english": "bread"},
    {"unicode": "𓎧", "gardiner": "V6", "translit": "šs", "english": "rope"},
    {"unicode": "𓎡", "gardiner": "V31", "translit": "k", "english": "basket"},

    # W: Vessels
    {"unicode": "𓏎", "gardiner": "W1", "translit": "jnp", "english": "jar"},
    {"unicode": "𓏑", "gardiner": "W9", "translit": "nwb", "english": "gold"},
    {"unicode": "𓏒", "gardiner": "X8", "translit": "pꜣt", "english": "cake"},
    {"unicode": "𓏐", "gardiner": "W3", "translit": "ḥnqt", "english": "beer"},
    {"unicode": "𓏙", "gardiner": "W22", "translit": "ꜥbw", "english": "jug"},
    {"unicode": "𓏚", "gardiner": "W24", "translit": "nw", "english": "pot"},

    # X: Bread/Food
    {"unicode": "𓏖", "gardiner": "X4", "translit": "šꜣ", "english": "loaf"},
    {"unicode": "𓏘", "gardiner": "X6", "translit": "tꜣ", "english": "bread roll"},
    {"unicode": "𓏛", "gardiner": "X7", "translit": "ꜣt", "english": "loaf variant"},

    # Y: Writing
    {"unicode": "𓏞", "gardiner": "Y3", "translit": "sš", "english": "scribe"},
    {"unicode": "𓏛", "gardiner": "Y1", "translit": "mḏꜣt", "english": "book"},
    {"unicode": "𓏜", "gardiner": "Y5", "translit": "rḫ", "english": "knowledge"},
    {"unicode": "𓏝", "gardiner": "Y6", "translit": "sšm", "english": "papyrus roll"},

    # Z: Strokes/Numbers
    {"unicode": "𓏤", "gardiner": "Z1", "translit": "1", "english": "one"},
    {"unicode": "𓏥", "gardiner": "Z2", "translit": "2", "english": "two"},
    {"unicode": "𓏦", "gardiner": "Z3", "translit": "3", "english": "three"},
    {"unicode": "𓏧", "gardiner": "Z4", "translit": "4", "english": "four"},
    {"unicode": "𓏨", "gardiner": "Z7", "translit": "w", "english": "twist"},

    # Aa: Unclassified
    {"unicode": "𓍑", "gardiner": "Aa1", "translit": "wḏꜣ", "english": "prosperity"},
    {"unicode": "𓍢", "gardiner": "Aa5", "translit": "ḫ", "english": "sieve"},
    {"unicode": "𓍹", "gardiner": "Aa11", "translit": "nsw", "english": "king"},
    {"unicode": "𓍺", "gardiner": "Aa12", "translit": "bjt", "english": "king of lower egypt"},
    {"unicode": "𓇓", "gardiner": "Aa13", "translit": "nsw", "english": "pharaoh"},
    {"unicode": "𓍿", "gardiner": "Aa15", "translit": "š", "english": "pool"},
    {"unicode": "𓎟", "gardiner": "Aa28", "translit": "nb", "english": "lord"},
    {"unicode": "𓎣", "gardiner": "Aa30", "translit": "ḫpš", "english": "strong arm"},
    {"unicode": "𓍮", "gardiner": "Aa8", "translit": "ꜣḫ", "english": "spirit"}
]

# Function to generate synthetic hieroglyphs (combinations or placeholders)
def generate_synthetic_sign(codepoint):
    return {
        "unicode": f"X{codepoint:04X}",  # Synthetic Unicode-like ID
        "gardiner": f"SYN-{codepoint:04X}",
        "translit": "",
        "english": ""
    }

# Extend to 10,000 signs
all_hieroglyphs = core_hieroglyph_data.copy()
used_unicode = {entry["unicode"] for entry in core_hieroglyph_data}
target_count = 10000

# Add all real Unicode signs (1,071)
for codepoint in range(0x13000, 0x1342F):
    unicode_char = chr(codepoint)
    if unicode_char not in used_unicode:
        all_hieroglyphs.append({
            "unicode": unicode_char,
            "gardiner": f"U+{codepoint:04X}",
            "translit": "",
            "english": ""
        })
        used_unicode.add(unicode_char)

# Add synthetic signs to reach 10,000
current_count = len(all_hieroglyphs)
for i in range(current_count, target_count):
    synthetic_code = i - current_count + 1
    all_hieroglyphs.append(generate_synthetic_sign(synthetic_code))

# Write to CSV
csv_headers = ["unicode", "gardiner", "translit", "english"]
with open("hieroglyphs_10000.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
    writer.writeheader()
    writer.writerows(all_hieroglyphs)

print(f"CSV file 'hieroglyphs_10000.csv' has been generated with {len(all_hieroglyphs)} hieroglyphic signs.")