import csv, json
from collections import Counter

big, startup = Counter(), Counter()
big_n = startup_n = 0

with open('data/analysis/clusters.csv', encoding='utf-8') as f:
    for r in csv.DictReader(f):
        techs = set(t.strip() for t in r['tech_all'].split(',') if t.strip())
        if r['회사규모'] == '대기업':
            big_n += 1
            for t in techs: big[t] += 1
        elif r['회사규모'] == '스타트업/중소':
            startup_n += 1
            for t in techs: startup[t] += 1

# 양쪽 합산 기준 상위 10개 기술
alltech = Counter()
for t, c in big.items():     alltech[t] += c
for t, c in startup.items(): alltech[t] += c
top = [t for t, _ in alltech.most_common(10)]

result = {
    'counts': {'대기업': big_n, '스타트업/중소': startup_n},
    'tech': [
        {
            '기술': t,
            '대기업': round(big[t] / big_n * 100, 1),
            '스타트업': round(startup[t] / startup_n * 100, 1)
        }
        for t in top
    ]
}

with open('data/analysis/company_size.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print('생성 완료: data/analysis/company_size.json')