import json
import bs4 as bs
import requests


PAGE_DIR = "pages/Software design pattern - Wikipedia.html"
LEVEL = 4
LEVEL_LABEL = "Design Patterns"

soup = bs.BeautifulSoup(open(PAGE_DIR), "html.parser")
tables = soup.find_all("table", {"class": "wikitable"})
titles = soup.find_all("h3")

json_data = {
    "level": LEVEL,
    "label": LEVEL_LABEL,
    "concepts": []
}


for table, title in zip(tables, titles):
    lines = table.find_all("tr")
    description = title.find_next("p")
    json_data["concepts"].append({
        "name": title.text.strip(),
        "description": description.get_text().strip() if description else "",
        "subconcepts": []
    })
    for line in lines:
        cells = line.find_all("td")
        subconcept = {}
        if len(cells) < 2:
            continue
        subconcept["name"] = cells[0].text.strip()
        subconcept["description"] = cells[1].text.strip()
        json_data["concepts"][-1]["subconcepts"].append(subconcept)

with open("data/concepts/level_4.json", "w") as f:
    json.dump(json_data, f)