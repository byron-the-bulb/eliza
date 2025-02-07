import { embed } from "./embedding.ts";
import { splitChunks } from "./generation.ts";
import elizaLogger from "./logger.ts";
import {
    IAgentRuntime,
    IRAGKnowledgeManager,
    RAGKnowledgeItem,
    UUID,
    ModelClass
} from "./types.ts";
import { stringToUuid } from "./uuid.ts";
import { ZeroEntropy }  from 'zeroentropy';
import { generateText } from "./generation.ts";




/**
 * Manage knowledge in the database using the ZeroEntropy API.
 */
export class ZeroEntropyRAGKnowledgeManager implements IRAGKnowledgeManager {
    /**
     * The AgentRuntime instance associated with this manager.
     */
    runtime: IAgentRuntime;

    /**
     * The name of the database table this manager operates on.
     */
    tableName: string;

    zclient: ZeroEntropy;

    /**
     * Constructs a new KnowledgeManager instance.
     * @param opts Options for the manager.
     * @param opts.tableName The name of the table this manager will operate on.
     * @param opts.runtime The AgentRuntime instance associated with this manager.
     */
    constructor(opts: { tableName: string; runtime: IAgentRuntime }) {
        this.runtime = opts.runtime;
        elizaLogger.info("Initializing ZeroEntropyRAGKnowledgeManager");
        this.zclient = new ZeroEntropy(
            {
                apiKey: this.runtime.character.settings?.secrets?.["ZEROENTROPY_API_KEY"] || process.env["ZEROENTROPY_API_KEY"]
            }
        );
    }

    private readonly defaultRAGMatchThreshold = 0.85;
    private readonly defaultRAGMatchCount = 5;

    /**
     * Common English stop words to filter out from query analysis
     */
    private readonly stopWords = new Set([
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "his",
        "how",
        "hey",
        "i",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
        "would",
        "there",
        "their",
        "they",
        "your",
        "you",
    ]);

    /**
     * Filters out stop words and returns meaningful terms
     */
    private getQueryTerms(query: string): string[] {
        return query
            .toLowerCase()
            .split(" ")
            .filter((term) => term.length > 3) // Filter very short words
            .filter((term) => !this.stopWords.has(term)); // Filter stop words
    }

    /**
     * Preprocesses text content for better RAG performance.
     * @param content The text content to preprocess.
     * @returns The preprocessed text.
     */

    private preprocess(content: string): string {
        if (!content || typeof content !== "string") {
            elizaLogger.warn("Invalid input for preprocessing");
            return "";
        }

        return content
            .replace(/```[\s\S]*?```/g, "")
            .replace(/`.*?`/g, "")
            .replace(/#{1,6}\s*(.*)/g, "$1")
            .replace(/!\[(.*?)\]\(.*?\)/g, "$1")
            .replace(/\[(.*?)\]\(.*?\)/g, "$1")
            .replace(/(https?:\/\/)?(www\.)?([^\s]+\.[^\s]+)/g, "$3")
            .replace(/<@[!&]?\d+>/g, "")
            .replace(/<[^>]*>/g, "")
            .replace(/^\s*[-*_]{3,}\s*$/gm, "")
            .replace(/\/\*[\s\S]*?\*\//g, "")
            .replace(/\/\/.*/g, "")
            .replace(/\s+/g, " ")
            .replace(/\n{3,}/g, "\n\n")
            .replace(/[^a-zA-Z0-9\s\-_./:?=&]/g, "")
            .trim()
            .toLowerCase();
    }

    private hasProximityMatch(text: string, terms: string[]): boolean {
        const words = text.toLowerCase().split(" ");
        const positions = terms
            .map((term) => words.findIndex((w) => w.includes(term)))
            .filter((pos) => pos !== -1);

        if (positions.length < 2) return false;

        // Check if any matches are within 5 words of each other
        for (let i = 0; i < positions.length - 1; i++) {
            if (Math.abs(positions[i] - positions[i + 1]) <= 5) {
                return true;
            }
        }
        return false;
    }

    async generateRAGQuery(
        runtime: IAgentRuntime,
        query: string,
        conversationContext: string
      ): Promise<string> {
        const prompt = `Your task is to generate a query for a RAG system given the following context:
      <message>
      ${query}
      </message>
      <previous conversation>
      ${conversationContext}
      </previous conversation>

      The RAG query should cover all relevant subjects and concepts included in the context, prioritizing the retrieval of content most relevant to the message.
      Return the text of the query only.`;

        const llmResponse = await generateText({
          runtime,
          context: prompt,
          modelClass: ModelClass.MEDIUM,
        });

        return llmResponse.trim();
      }

    async getKnowledge(params: {
        query?: string;
        id?: UUID;
        conversationContext?: string;
        limit?: number;
        agentId?: UUID;
    }): Promise<RAGKnowledgeItem[]> {
        const agentId = params.agentId || this.runtime.agentId;

        elizaLogger.info("Getting knowledge for agentId : " + agentId);
        elizaLogger.info("Query : [" + params.query + "] id : [" + params.id + "]");

        const collectionName = this.runtime.agentId;
        if (params.id) {
            try {
                const response = await this.zclient.documents.getInfo({
                    collection_name: collectionName,
                    path: params.id,
                });
                elizaLogger.info("Knowledge found for id : " + params.id);
                return [{
                    id: stringToUuid(response.document.path),
                    agentId: agentId,
                    content: { text: response.document.content },
                }];
            } catch (error) {
                elizaLogger.info(`Knowledge ${params.id} not found:`, error);
            }
        }

        // If no id or no direct results, perform semantic search
        if (params.query) {
            try {
                const processedQuery = this.preprocess(params.query);
                let searchQuery = processedQuery;
                if (params.conversationContext) {
                    const relevantContext = this.preprocess(
                        params.conversationContext
                    );
                    searchQuery = `Primary Query: ${processedQuery} \n Context: ${relevantContext}`;
                }

                searchQuery = await this.generateRAGQuery(this.runtime, processedQuery, params.conversationContext);
                elizaLogger.info("ZE Knowledge search query : " + searchQuery);
                const response = await this.zclient.queries.topSnippets({
                    collection_name: collectionName,
                    query: searchQuery,
                    k: params.limit || this.defaultRAGMatchCount,
                    precise_responses: false,
                });
                elizaLogger.info("ZE Knowledge search results : " + response.results);

                return response.results.map((result) => ({
                    id: stringToUuid(result.path),
                    agentId: agentId, //need to query metadata for agentId
                    score: result.score,
                    content: {
                        text: result.content,
                    },
                }));
            } catch (error) {
                elizaLogger.error(`[RAG Search Error] ${error}`);
                return [];
            }
        }

        // If neither id nor query provided, return empty array
        return [];
    }

    async createKnowledge(item: RAGKnowledgeItem): Promise<void> {
        if (!item.content.text) {
            elizaLogger.warn("Empty content in knowledge item");
            return;
        }

        const collectionName = this.runtime.agentId;
        elizaLogger.info("Creating collection : " + collectionName);
        try {
            await this.zclient.collections.add({
                collection_name: collectionName,
            });
            elizaLogger.info(`Collection '${collectionName}' created successfully.`);
        } catch (err) {
            // It's possible the collection already exists.
            elizaLogger.info(`Collection '${collectionName}' may already exist. Proceeding...`);
        }

        try {
            elizaLogger.info("Adding direct knowledge text to ZeroEntropy : " + item.content.text.substring(0, 15)+"...");
            // Process main document
            const processedContent = this.preprocess(item.content.text);
            await this.zclient.documents.add({
                collection_name: collectionName,
                path: item.id,
                content: { type: "text", text: processedContent },
            });

            let indexed = false;
            while (!indexed) {
                const status = await this.zclient.documents.getInfo({
                    collection_name: collectionName,
                    path: "docs/"+item.id+".txt",
                });

                if (status.document.index_status === "indexed") {
                    elizaLogger.info("Direct knowledge text is indexed.");
                    indexed = true;
                } else {
                    await new Promise(resolve => setTimeout(resolve, 500));
                }
            }

        } catch (error) {
            elizaLogger.error(`Error processing knowledge ${item.id}:`, error);
            throw error;
        }
    }

    async searchKnowledge(params: {
        agentId: UUID;
        embedding: Float32Array | number[];
        match_threshold?: number;
        match_count?: number;
        searchText?: string;
    }): Promise<RAGKnowledgeItem[]> {
        const collectionName = this.runtime.agentId;
        elizaLogger.info("Searching ZeroEntropy for : " + params.searchText);
        const response = await this.zclient.queries.topSnippets({
            collection_name: collectionName,
            query: params.searchText,
            k: params.match_count || this.defaultRAGMatchCount,
        });
        elizaLogger.info("Search results : " + response.results);
        return response.results.map((result) => ({
            id: stringToUuid(result.path),
            agentId: params.agentId, //need to query metadata for agentId
            score: result.score,
            content: {
                text: result.content,
            },
        }));
    }

    async removeKnowledge(id: UUID): Promise<void> {
        await this.runtime.databaseAdapter.removeKnowledge(id);
    }

    async clearKnowledge(shared?: boolean): Promise<void> {
        await this.runtime.databaseAdapter.clearKnowledge(
            this.runtime.agentId,
            shared ? shared : false
        );
    }

    async processFile(file: {
        path: string;
        content: string;
        type: "pdf" | "md" | "txt";
        isShared?: boolean;
    }): Promise<void> {
        const startTime = Date.now();
        elizaLogger.info(`[File Progress] Starting ${file.path}`);

        try {
            const collectionName = this.runtime.agentId;

            // Ensure the collection exists.
            try {
                await this.zclient.collections.add({
                    collection_name: collectionName,
                });
                elizaLogger.info(`Collection '${collectionName}' created successfully.`);
            } catch (err) {
                // It's possible the collection already exists.
                elizaLogger.info(`Collection '${collectionName}' may already exist. Proceeding...`);
            }

            // Prepare the content payload using the proper format.
            // PDFs are converted to Base64; text-based files are sent as plain text.
            let contentPayload: { type : any, base64_data : string} | { type : any, text : string};
            if (file.type === "pdf") {
                const base64Content = Buffer.from(file.content).toString("base64");
                contentPayload = {
                    type: "auto",
                    base64_data: base64Content,
                } ;
            } else {
                contentPayload = {
                    type: "text",
                    text: file.content,
                };
            }
            elizaLogger.info("Adding document to ZeroEntropy : " + file.path);
            //elizaLogger.info("Content payload : " + contentPayload);
            // Add the document to ZeroEntropy's RAG storage.
            const response = await this.zclient.documents.add({
                collection_name: collectionName,
                path: stringToUuid(file.path),
                content: contentPayload,
                metadata: {
                    timestamp: new Date().toISOString(),
                    fileSizeKB: (new TextEncoder().encode(file.content).length / 1024).toFixed(2) + " KB",
                    file_type: file.type,
                    agentId: this.runtime.agentId,
                },
            });

            const totalTime = (Date.now() - startTime) / 1000;
            elizaLogger.info(`[Complete] Processed ${file.path} in ${totalTime.toFixed(2)}s`);
            elizaLogger.info(response.message);
        } catch (error) {
            if (
                file.isShared &&
                error?.code === "SQLITE_CONSTRAINT_PRIMARYKEY"
            ) {
                elizaLogger.info(
                    `Shared knowledge ${file.path} already exists in storage, skipping creation`
                );
                return;
            }
            elizaLogger.error(`Error processing file ${file.path}:`, error);
            throw error;
        }
    }
}
