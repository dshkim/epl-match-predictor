import requests
import pandas as pd
from bs4 import BeautifulSoup
import time

def fetch_soup(url):
    """Fetches the HTML content of a URL and parses it with BeautifulSoup."""
    response = requests.get(url)
    time.sleep(15)  # Respectful delay between requests
    return BeautifulSoup(response.text, 'html.parser')

def get_team_urls(year):
    """Extracts team URLs from the Premier League stats page for a given year."""
    url = f"https://fbref.com/en/comps/9/{year-1}-{year}/{year-1}-{year}-Premier-League-Stats"
    soup = fetch_soup(url)
    
    # Find the standings table
    standings_table = soup.select('table.stats_table')[0]
    
    # Extract and filter links to get team stats pages only
    links = [l.get("href") for l in standings_table.find_all('a')]
    team_urls = [f"https://fbref.com{l}" for l in links if '/squads/' in l]
    
    return team_urls

def get_team_data(team_url):
    """Extracts match and shooting data for a given team's stats page."""
    team_name = team_url.split("/")[-1].replace("-Stats", "").replace("-", " ")
    matches_url = team_url
    matches_soup = fetch_soup(matches_url)
    
    # Read match data
    matches = pd.read_html(matches_soup.prettify(), match="Scores & Fixtures")[0]
    
    # Find and fetch shooting stats
    links = [l.get("href") for l in matches_soup.find_all('a')]
    shooting_links = [l for l in links if l and 'all_comps/shooting/' in l]
    shooting_url = f"https://fbref.com{shooting_links[0]}"
    shooting_soup = fetch_soup(shooting_url)
    
    # Read shooting data
    shooting = pd.read_html(shooting_soup.prettify(), match="Shooting")[0]
    shooting.columns = shooting.columns.droplevel()
    
    return team_name, matches, shooting

def process_team_data(matches, shooting, team_name):
    """Merges match and shooting data and filters for Premier League matches."""
    try:
        team_data = matches.merge(
            shooting[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]],
            on="Date"
        )
    except ValueError:
        return None
    
    team_data = team_data[team_data["Comp"] == "Premier League"]
    return team_data

def web_scraper():
    """Main function to scrape and process EPL data for the specified years."""
    years = list(range(2020, 2025))
    all_matches = []

    for year in years:
        team_urls = get_team_urls(year)
        
        for team_url in team_urls:
            team_name, matches, shooting = get_team_data(team_url)
            team_data = process_team_data(matches, shooting, team_name)
            
            if team_data is not None:
                team_data["Season"] = year
                team_data["Team"] = team_name
                all_matches.append(team_data)
                time.sleep(15)  # Respectful delay between requests
    
    # Combine all matches data into a single DataFrame
    all_matches_df = pd.concat(all_matches, ignore_index=True)
    
    #Standardize capitalization for all columns
    all_matches_df.columns = [c.lower() for c in all_matches_df.columns]
    
    # Save the DataFrame to a CSV file
    all_matches_df.to_csv('epl_matches_2020_2024.csv', index=False)

web_scraper()
