#take the tables from https://pra.ufpr.br/ru/ru-central/ and put them in a folder called "cardapio"

import os
import sys
import requests
from bs4 import BeautifulSoup

cardapio_url = "https://pra.ufpr.br/ru/ru-central/"
cardapio_folder = "cardapio"
cardapio_file = "cardapio.html"
cardapio_file_path = os.path.join(cardapio_folder, cardapio_file)
cardapio_file_url = os.path.join(cardapio_url, cardapio_file)