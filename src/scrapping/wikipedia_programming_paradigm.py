import json
import re

import bs4 as bs


PAGE_DIR = "pages/Programming paradigm - Wikipedia.html"
OUTPUT_PATH = "data/concepts/level_3.json"
LEVEL = 3
LEVEL_LABEL = "Paradigm and Algorithmic Logic level"


def clean_text(text):
    text = re.sub(r"\[\d+\]", "", text)
    return " ".join(text.split()).capitalize()


def split_title_description(text):
    parts = re.split(r"\s+[-–—]\s+", clean_text(text), maxsplit=1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]


def direct_li_text(li_tag):
    parts = []

    for child in li_tag.children:
        if getattr(child, "name", None) == "ul":
            continue
        if getattr(child, "get_text", None):
            parts.append(child.get_text(" ", strip=True))
        else:
            parts.append(str(child).strip())

    return clean_text(" ".join(part for part in parts if part))


def parse_list_item(li_tag):
    name, description = split_title_description(direct_li_text(li_tag))
    concept = {
        "name": name,
        "description": description.capitalize(),
    }

    subconcepts = []
    for nested_list in li_tag.find_all("ul", recursive=False):
        subconcepts.extend(parse_list(nested_list))

    if subconcepts:
        concept["subconcepts"] = subconcepts

    return concept


def parse_list(ul_tag):
    concepts = []
    for li_tag in ul_tag.find_all("li", recursive=False):
        concepts.append(parse_list_item(li_tag))
    return concepts


def find_programming_paradigms_list(soup):
    for ul_tag in soup.find_all("ul"):
        first_item = ul_tag.find("li", recursive=False)
        if not first_item:
            continue

        name, _ = split_title_description(direct_li_text(first_item))
        if name == "Imperative":
            return ul_tag

    raise ValueError("Programming paradigms list not found")


json_data = {
    "level": LEVEL,
    "label": LEVEL_LABEL,
    "concepts": [],
}

soup = bs.BeautifulSoup(open(PAGE_DIR), "html.parser")
ul_tag = find_programming_paradigms_list(soup)
json_data["concepts"] = parse_list(ul_tag)

with open(OUTPUT_PATH, "w") as f:
    json.dump(json_data, f, indent=2)
