import scrapy
import datetime
import itertools

class ScraperSpider(scrapy.Spider):
    name = "scraper"
    allowed_domains = ["en.wikipedia.org"]

    months = ['January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December']

    latest_month = datetime.datetime.now().month   # 1-indexed
    latest_year  = datetime.datetime.now().year

    start_urls = []
    for year, month in itertools.product(range(2025, latest_year + 1), range(12)):
        if year < latest_year or (year == latest_year and month < latest_month):
            start_urls.append(
                f"https://en.wikipedia.org/wiki/Portal:Current_events/{months[month]}_{year}"
            )
        else:
            break

    def parse(self, response):
        current_news = response.css("div.current-events")

        for i in current_news:
            yield {
                
                "topic": i.css(".current-events-content.description p b *::text").getall(),
                "date": i.css(".current-events-title *::text").get(),
                ## problem here is that it collects ALL of the text data
                # I can include a delimiter than be used to separate the text
                # i can include the heading
                "text": i.css("ul li *::text").getall(),
                "headings": i.css("div.current-events-content.description.current-events-content-heading")
                # 
            }