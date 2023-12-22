mkdir -p data
wget "https://opendata.vancouver.ca/api/explore/v2.1/catalog/datasets/property-tax-report/exports/csv?lang=en&timezone=America%2FLos_Angeles&use_labels=true&delimiter=%3B" -O data/property-tax-report.csv
