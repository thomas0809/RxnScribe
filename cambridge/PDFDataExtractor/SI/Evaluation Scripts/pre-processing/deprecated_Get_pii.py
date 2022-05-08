'''
This script is used to get journal piis
'''
import numpy as np
import json
import requests
import time


query = 'solar'
base_url = 'https://api.elsevier.com/content/search/sciencedirect'

data = {"qs": query,
        "date": 0,
        "volume": 1,
        "display": {"show": 100, "offset": 0},
        "filter": {"openAccess": 'true'},
        # "pub": "Solar Energy Materials and Solar Cells"
        }


headers = {'X-ELS-APIKey': 'b8edec6af4a9689879e5f5abcf4ae282',
           'Content-Type': 'application/json',
           'Accept': 'application/json'}


def get_response(url, data, headers):
    response = requests.put(url, data=json.dumps(data), headers=headers)
    response = response.text.replace('false', 'False').replace('true', 'True')

    try:
        response = eval(response)
    except:
        print(response)

    return response


def get_piis(data, headers):
    piis = []
    year_count = 0

    for volume in np.arange(1, 100):
        print('volume', volume)
        data['volume'] = int(volume)

        for year in np.arange(2013, 2021)[::-1]:
            print(year)
            time.sleep(5)

            data["date"] = int(year)
            response = get_response(base_url, data, headers)

            if 'resultsFound' in response.keys():
                print('results found: ', response['resultsFound'])
                n = int(np.ceil(response['resultsFound'] / 100))

            else:
                n = 60

            for offset in range(n + 1):
                data["display"]["offset"] = offset
                time.sleep(5)
                response = get_response(base_url, data, headers)

                if 'results' in response.keys():
                    results = response['results']
                    for result in results:
                        if 'pii' in result:
                            piis.append(result['pii'])

                            year_count += 1
                            with open(r'./pii/{}.json'.format(year), 'a', encoding='utf-8') as json_file:
                                json.dump(result['pii'], json_file, ensure_ascii=False)
                                json_file.write('\n')

        print('- V: {} '.format(volume) + 'done')
        print(len(piis))

    return piis

get_piis(data, headers)