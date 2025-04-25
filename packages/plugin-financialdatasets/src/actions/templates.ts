export const getCompanyNewsTemplate = `Respond with a JSON object containing company ticker and date range for finacial news data.
Extract the company ticker from the most recent message. If no specific ticker is provided, respond with an error.
Extract the date range from the most recent message. If no specific date range is provided use the last 60 days from the current date and time.

The response must include:
- ticker: The company ticker
- fromDate: The start date in YYYY-MM-DD format
- toDate: The end date in YYYY-MM-DD format

{{providers}}

Example response:
\`\`\`json
{
    "ticker": "AAPL",
    "fromDate": "2024-01-01",
    "toDate": "2024-01-31"
}
}
\`\`\`
{{recentMessages}}
Extract the company ticker and date range from the most recent message.
Respond with a JSON markdown block containing both ticker and date range.`;