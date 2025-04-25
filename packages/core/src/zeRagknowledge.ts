import { embed } from "./embedding.ts";
import { splitChunks } from "./generation.ts";
import elizaLogger from "./logger.ts";
import crypto from 'crypto';
import {
    IAgentRuntime,
    IRAGKnowledgeManager,
    RAGKnowledgeItem,
    UUID,
    ModelClass,
    KnowledgeScope
} from "./types.ts";
import { stringToUuid } from "./uuid.ts";
import { ZeroEntropy }  from 'zeroentropy';
import { generateText } from "./generation.ts";
import { P } from "pino";




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
     * In-memory storage for document metadata
     */
    private documentMetadata: Map<string, Record<string, any>> = new Map();

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

        const collectionName = this.runtime.character.settings?.zeRagKnowledgeCollectionName || this.runtime.agentId;
        if (params.id) {
            try {
                const response = await this.zclient.documents.getInfo({
                    collection_name: collectionName,
                    path: params.id,
                    include_content: false
                });
                elizaLogger.info("Knowledge found for id : " + params.id);
                
                // Make sure we have valid content to avoid 'Cannot read properties of null' errors
                const documentContent = response.document.content || '';

                //if present add metadata to local memory
                if (response.document.metadata) {
                    this.documentMetadata.set(params.id, response.document.metadata);
                }
                
                // Note: We return raw content instead of formatted content
                // This is needed for proper comparison in runtime.ts
                
                // Get stored metadata which should include contentHash
                const storedMetadata = this.documentMetadata.get(params.id) || {};
                
                return [{
                    id: stringToUuid(response.document.path),
                    agentId: agentId,
                    content: { 
                        text: documentContent, 
                        metadata: { 
                            source: response.document.file_url,
                            contentHash: response.document.metadata?.contentHash
                        } 
                    },
                }];
            } catch (error) {
                elizaLogger.info(`Knowledge ${params.id} not found:`, error);
                return [];
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

                return response.results.map((result) => {
                    // Make sure we have valid content to avoid 'Cannot read properties of null' errors
                    const resultContent = result.content || '';
                    
                    // Format the content with metadata if available
                    let formattedContent = resultContent;
                    const metadata = this.documentMetadata.get(result.path);
                    if (metadata) {
                        formattedContent = this.formatContentWithMetadata(resultContent, metadata);
                    } else {
                        elizaLogger.warn(`Metadata not found for ${result.path}`);
                    }
                    
                    return {
                        id: stringToUuid(result.path),
                        agentId: agentId,
                        score: result.score,
                        content: {
                            text: formattedContent,
                        },
                    };
                });
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

        const collectionName = this.runtime.character.settings?.zeRagKnowledgeCollectionName || this.runtime.agentId;
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
                    path: item.id,
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
        const collectionName = this.runtime.character.settings?.zeRagKnowledgeCollectionName || this.runtime.agentId;
        elizaLogger.info("Searching ZeroEntropy for : " + params.searchText);
        const response = await this.zclient.queries.topSnippets({
            collection_name: collectionName,
            query: params.searchText,
            k: params.match_count || this.defaultRAGMatchCount,
            precise_responses: true,
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
        elizaLogger.info("Removing knowledge : " + id);
        try {
            const collectionName = this.runtime.character.settings?.zeRagKnowledgeCollectionName || this.runtime.agentId;
            await this.zclient.documents.delete({
                collection_name: collectionName,
                path: id,
            });
        } catch (error) {
            elizaLogger.warn(`Error removing knowledge ${id}:`, error);
        }
    }

    async clearKnowledge(shared?: boolean): Promise<void> {
        /*
        const collectionName = this.runtime.agentId;
        await this.zclient.collections.delete({
            collection_name: collectionName,
        });
        */
    }

    async cleanupDeletedKnowledgeFiles() {
        //throw new Error("Method not implemented.");
    }

    public generateScopedId(path: string, isShared: boolean): UUID {
        // Prefix the path with scope before generating UUID to ensure different IDs for shared vs private
        const scope = isShared ? KnowledgeScope.SHARED : KnowledgeScope.PRIVATE;
        const scopedPath = `${scope}-${path}`;
        return stringToUuid(scopedPath);
    }

    /**
     * Validates if metadata contains the required fields
     * @param metadata Metadata to validate
     * @param path Path of the document (for logging)
     * @returns true if valid, false otherwise
     */
    private validateMetadata(metadata: Record<string, any> | undefined, path: string): boolean {
        if (!metadata) {
            return false;
        }
        
        // Check for required fields
        const { title, author, type } = metadata;
        if (!title || !author || !type) {
            elizaLogger.warn(`Missing required metadata fields for ${path}. Required: title, author, type`);
            return false;
        }
        
        return true;
    }
    
    /**
     * Check if a document exists in Zero Entropy
     * @param documentPath Path of the document to check
     * @param isShared Whether the document is shared
     * @returns true if the document exists, false otherwise
     */
    async documentExists(documentPath: string, isShared?: boolean): Promise<boolean> {
        try {
            const collectionName = this.runtime.character.settings?.zeRagKnowledgeCollectionName || this.runtime.agentId;
            const scopedId = this.generateScopedId(documentPath, isShared);
            
            // Get the document info (don't need content for existence check)
            const response = await this.zclient.documents.getInfo({
                collection_name: collectionName,
                path: scopedId,
                include_content: false // No need for content when just checking existence
            });
            
            // If we get a response, the document exists
            return true;
        } catch (error) {
            // If we get an error, the document doesn't exist
            return false;
        }
    }
    

    /**
     * Calculate hash for content (Buffer or string)
     * @param content The content to hash
     * @returns SHA256 hash of the content
     */
    private calculateContentHash(content: Buffer | string): string {
        return crypto.createHash('sha256').update(content).digest('hex');
    }

    /**
     * Format content with metadata
     * @param content The raw content text
     * @param metadata The metadata to include
     * @returns Formatted content that includes metadata context
     */
    private formatContentWithMetadata(content: string, metadata: Record<string, any>): string {
        let metadataIntro = "";
        
        // Format depends on available metadata
        if (metadata.title && metadata.author) {
            if (metadata.type === 'book' || metadata.type === 'novel') {
                metadataIntro = `The ${metadata.type} "${metadata.title}" by ${metadata.author}, mentions the following: `;
            } else if (metadata.type === 'article' || metadata.type === 'paper') {
                metadataIntro = `The ${metadata.type} titled "${metadata.title}" by ${metadata.author}, states: `;
            } else if (metadata.type === 'report') {
                metadataIntro = `According to the report "${metadata.title}" by ${metadata.author}: `;
            } else {
                metadataIntro = `From "${metadata.title}" by ${metadata.author}: `;
            }
        } else if (metadata.title) {
            metadataIntro = `From "${metadata.title}": `;
        } else if (metadata.author) {
            metadataIntro = `Written by ${metadata.author}: `;
        }
        
        return metadataIntro + content;
    }

    /**
     * Update document metadata in ZeroEntropy
     * @param documentId Document ID (path)
     * @param metadata New metadata to apply
     */
    async updateDocumentMetadata(documentId: string, metadata: Record<string, any>): Promise<void> {
        try {
            const collectionName = this.runtime.character.settings?.zeRagKnowledgeCollectionName || this.runtime.agentId;
            
            // Check for required fields
            if (!this.validateMetadata(metadata, documentId)) {
                elizaLogger.warn(`Cannot update metadata for ${documentId}: missing required fields (title, author, type)`);
                return;
            }
            
            // Update the document metadata
            await this.zclient.documents.update({
                collection_name: collectionName,
                path: documentId,
                metadata: metadata
            });
            
            elizaLogger.info(`Updated metadata for document ${documentId}`);
        } catch (error) {
            elizaLogger.error(`Error updating metadata for document ${documentId}:`, error);
            throw error;
        }
    }

    async processFile(file: {
        path: string;
        content: Buffer | string;
        type: "pdf" | "md" | "txt";
        isShared?: boolean;
        metadata?: Record<string, any>;
    }): Promise<void> {
        const startTime = Date.now();
        elizaLogger.info(`[File Progress] Starting ${file.path}`);

        try {
            const collectionName = this.runtime.character.settings?.zeRagKnowledgeCollectionName || this.runtime.agentId;

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

            // Validate metadata
            if (!this.validateMetadata(file.metadata, file.path)) {
                elizaLogger.warn(`Skipping upload for ${file.path} - missing required metadata (title, author, type)`);
                return;
            }

            // Generate the scoped document ID
            const documentId = this.generateScopedId(file.path, file.isShared);
            
            // Check if document already exists
            const documentExists = await this.documentExists(file.path, file.isShared);
            
            // Prepare standard metadata
            const standardMetadata = {
                timestamp: new Date().toISOString(),
                fileSizeKB: (Buffer.isBuffer(file.content) 
                    ? file.content.length 
                    : new TextEncoder().encode(file.content as string).length / 1024).toFixed(2) + " KB",
                file_type: file.type,
                agentId: this.runtime.agentId,
            };
            
            // Calculate content hash
            const contentHash = this.calculateContentHash(file.content);
            
            // Combine standard metadata with document-specific metadata and content hash
            const fullMetadata = {
                ...standardMetadata,
                ...file.metadata,
                contentHash: contentHash
            };
            
            // Store metadata in memory - do this regardless of whether the document exists or is being created
            this.documentMetadata.set(documentId, fullMetadata);
            elizaLogger.info(`Stored metadata in memory for document ${file.path} ad ${documentId}`);
            
            if (documentExists) {
                elizaLogger.info(`Document ${file.path} already exists, updating metadata...`);
                
                try {
                    // Update the metadata
                    await this.updateDocumentMetadata(documentId, fullMetadata);
                    elizaLogger.info(`Updated metadata for existing document ${file.path}`);
                } catch (updateErr) {
                    elizaLogger.error(`Failed to update metadata for existing document ${file.path}:`, updateErr);
                    throw updateErr;
                }
                
                const totalTime = (Date.now() - startTime) / 1000;
                elizaLogger.info(`[Complete] Updated ${file.path} in ${totalTime.toFixed(2)}s`);
                return;
            }

            // If we get here, the document doesn't exist yet, so we'll create it
            
            // Prepare the content payload using the proper format.
            // PDFs are converted to Base64; text-based files are sent as plain text.
            let contentPayload: { type : any, base64_data : string} | { type : any, text : string};
            if (file.type === "pdf") {
                // If content is already a Buffer, use it directly; otherwise, convert to Buffer
                const buffer = Buffer.isBuffer(file.content) ? file.content : Buffer.from(file.content);
                const base64Content = buffer.toString("base64");
                contentPayload = {
                    type: "auto",
                    base64_data: base64Content,
                } ;
            } else {
                // For text files, convert Buffer to string if needed
                const textContent = Buffer.isBuffer(file.content) 
                    ? file.content.toString('utf-8') 
                    : file.content;
                contentPayload = {
                    type: "text",
                    text: textContent,
                };
            }
            elizaLogger.info("Adding document to ZeroEntropy : " + file.path+" as " + documentId);
            
            // Add the document to ZeroEntropy's RAG storage with full metadata.
            const response = await this.zclient.documents.add({
                collection_name: collectionName,
                path: documentId,
                content: contentPayload,
                metadata: fullMetadata,
            });

            const totalTime = (Date.now() - startTime) / 1000;
            elizaLogger.info(`[Complete] Processed ${file.path} in ${totalTime.toFixed(2)}s`);
            elizaLogger.info(response.message);
        } catch (error) {
            elizaLogger.error(`Error processing file ${file.path}:`, error);
            throw error;
        }

    }
}
