import {
  ActionExample,
  Content,
  generateText,
  HandlerCallback,
  IAgentRuntime,
  Memory,
  ModelClass,
  State,
  type Action,
  elizaLogger,
  composeContext,
  generateMessageResponse
} from "@elizaos/core";

import axios from 'axios';
import { getCompanyNewsTemplate } from "./templates"
import { getCompanyNewsExamples } from "./examples"

export const getCompanyNews : Action = {
  name: "GET_COMPANY_NEWS",
  similes: ["GET_COMPANY_NEWS", "GET_FINANCIAL_NEWS", "GET_CURRENT_FINANCIAL_NEWS"],
  description: "Fetches news for a given company",
  validate: async (_runtime: IAgentRuntime, _message: Memory) => {
    const apiKey = process.env.FINANCIAL_DATASETS_API_KEY;
    if (!apiKey) {
      throw new Error('FINANCIAL_DATASETS_API_KEY environment variable is not set');
    }
    return true;
  },
  examples : getCompanyNewsExamples as ActionExample[][],
  handler: async (
    _runtime: IAgentRuntime,
    _message: Memory,
    _state: State,
    _options: { [key: string]: unknown; },
    _callback: HandlerCallback,
  ) :Promise<boolean> => {
      let currentState: State = _state;
      if (!currentState) {
        currentState = (await _runtime.composeState(_message)) as State;
      }
      currentState = await _runtime.updateRecentMessageState(currentState);

      const companyNewsContext = composeContext({
        state: currentState,
        template: getCompanyNewsTemplate,
      });

      const queryContent = await generateMessageResponse({
        runtime: _runtime,
        context: companyNewsContext,
        modelClass: ModelClass.SMALL,
      });

      if (!(queryContent?.ticker && queryContent?.fromDate && queryContent?.toDate && !queryContent?.error)) {
        elizaLogger.error('Failed to retrieve company name and date range', queryContent);
        return false;
      }

      try {
        const response = await axios.get('https://api.financialdatasets.ai/news', {
          headers: {
                'X-API-KEY': process.env.FINANCIAL_DATASETS_API_KEY
          },
          params: {
                ticker: queryContent?.ticker,
                start_date: queryContent?.fromDate,
                end_date: queryContent?.toDate,
                limit: 10
          }
        });
        elizaLogger.info('Fetched company news for ', queryContent?.ticker, 'from ', queryContent?.fromDate, 'to ', queryContent?.toDate);
        //console.log("Response: ", response);

        if (_callback) {
          const ret_content = response.data?.news.map((article, index) => {
            return [
                `📰 Article ${index + 1} - ${article.date}`,
                '━━━━━━━━━━━━━━━━━━━━━━',
                `📌 **${article.title || 'No title'}**\n`,
                `📝 ${article.author} @ ${article.source}\n`,
                `Sentiment: ${article.sentiment}\n`,
                `🔗 Read article : ${article.url}\n\n`
            ].join('\n');
          }).join('\n');
          console.log("Content: ", ret_content);
          _callback({
            text : `The news articles for ${queryContent?.ticker} from ${queryContent?.fromDate} to ${queryContent?.toDate} are:\n${ret_content}`,
          });
          return true;
        };
    } catch (error) {
      elizaLogger.error('Failed to fetch company news', error);
      return false;
    }
    return false;
  },
};
