from linkedin_jobs_scraper import LinkedinScraper
from linkedin_jobs_scraper.query import Query, QueryOptions
from linkedin_jobs_scraper.events import Events, EventData
import pandas as pd

def scrape_linkedin_jobs(keyword, location, limit=50):
    results = []
    scraper = LinkedinScraper(headless=True, max_workers=2)

    @scraper.on(Events.DATA)
    def on_data(data: EventData):
        results.append({
            'Job Title': data.title,
            'Job Description': data.description,
            'Company': data.company,
            'Link': data.link
        })

    @scraper.on(Events.END)
    def on_end():
        print(f"Scraped {len(results)} jobs.")

    scraper.run([Query(keyword, location, options=QueryOptions(limit=limit))])
    return pd.DataFrame(results)

if __name__ == "__main__":
    df = scrape_linkedin_jobs("Data Scientist", "India", limit=30)
    df.to_csv("job_descriptions.csv", index=False)
    print("Saved jobs to CSV.")
