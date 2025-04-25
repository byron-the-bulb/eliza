import { Plugin } from "@elizaos/core";
import { getCompanyNews } from './actions/company-news';

export const financialDatasetsPlugin: Plugin = {
    name: 'financialDatasetsPlugin',
    description: 'Access financial datasets from FinancialDatasets.ai API',
    actions: [getCompanyNews],
};

export default financialDatasetsPlugin;
