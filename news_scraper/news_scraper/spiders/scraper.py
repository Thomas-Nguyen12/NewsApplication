import scrapy
import datetime

class ScraperSpider(scrapy.Spider):
    name = "scraper"
    allowed_domains = ["en.wikipedia.org"]#   
    
    #  I can manipulate the start url here to access multiple months
    # Alternatively, I can create a loop that adds years and months from now until last year 
    start_urls = []
    
    months = ['January', 'February', 'March', "April", 'May', 'June', 'July', 'August',
              'September', 'October', 'November', 'December']
    

    latest_month = datetime.datetime.now().month 
    latest_year = datetime.datetime.now().year 
    years = [i for i in range(2025, latest_year + 1)]
    
    
    print (f"Years: {years}") 
    
    
    
    
    # the loop will be exited once it reaches the current date
    

    for year in years: 
        for month in range(len(months)): 
            # This will iterate through months first for each year
            # I can convert the month into a numerical format and compare it to the latest month
            if year <= latest_year and month <= latest_month:
                start_urls.append(f"https://en.wikipedia.org/wiki/Portal:Current_events/{months[month]}_{year}")
            else: 
                # breaking the loop
                break
    
    # I can then remove duplicate dates (as these will contain the same news reports)
    # Maybe I can include a delimiter to separate the topics 
    def parse(self, response):
        """ 
        selecting the headlines and their summaries, placing them into a csv file
        
        To return the results in a csv format, enter the following code: 
        scray crawl scraper -O output.csv 
        
        
        I can collect news reports from all months of this year by manipulating the url, for example: 
        https://en.wikipedia.org/wiki/Portal:Current_events/April_2025
        https://en.wikipedia.org/wiki/Portal:Current_events/May_2025
        """

        # summary will return multiple date 
        
        # there will be a lot here
        # collecting all
        current_news = response.css("div.current-events")
        
        """ 
        I need to store the data somehow and specify a structure
        
        dataset structure:
        
        columns: 
        1. date
        """
        
        for i in current_news: 
            
            # forating the data
            
            ## yield generates the dataset 
            """
            
            I can include some delimiters for each row 
            """
            
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
