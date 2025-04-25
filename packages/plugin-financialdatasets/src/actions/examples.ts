import type { ActionExample } from "@elizaos/core";

export const getCompanyNewsExamples: ActionExample[][] = [
    [
        {
            user: "{{user1}}",
            content: {
                text: "What are the latest financial news",
            },
        },
        {
            user: "{{agent}}",
            content: {
                text: "For which company?",
            },
        },
        {
            user: "{{user1}}",
            content: {
                text: "Apple",
            },
        },
        {
            user: "{{agent}}",
            content: {
                text: "Let me retrieve the financial news for Apple in the last 60 days.",
                action: "GET_COMPANY_NEWS",
            },
        },
    ],
    [
        {
            user: "{{user1}}",
            content: {
                text: "What are the financial news for Google in the last week?",
            },
        },
        {
            user: "{{agent}}",
            content: {
                text: "I'll check the financial news for Google in the last week for you.",
                action: "GET_COMPANY_NEWS",
            },
        }
    ],
    [
        {
            user: "{{user1}}",
            content: {
                text: "What is the sentiment of the financial news for AAPL in the last 60 days?",
            },
        },
        {
            user: "{{agent}}",
            content: {
                text: "I'll check the sentiment of the financial news for AAPL in the last 60 days.",
                action: "GET_COMPANY_NEWS",
            },
        }
    ],
];