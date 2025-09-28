import { generateId } from './random';
import pdfParse from "pdf-parse";
import { client } from "./qdrant";
import { EMBEDDING_MODEL_NAME } from '../data/ai-config';
import { createMistral } from '@ai-sdk/mistral';
import { embed } from 'ai';

// For embedding 
const mistral = createMistral({
    apiKey: process.env.MISTRAL_EMBEDDING_API_KEY
})

export const chunkDocument = async (pdfBuffer: ArrayBuffer) => {
    const buffer = Buffer.from(pdfBuffer);

    const data = await pdfParse(buffer);
    const text = data.text;

    const sentences = text
        .split(/(?<=[.!?])\s+/)
        .map(sentence => sentence.trim())
        .filter(Boolean);

    const chunks: string[] = [];
    for (let i = 0; i < sentences.length; i += 1) {
        const chunk = sentences.slice(i, i + 3).join(" ");
        chunks.push(chunk);
    }

    return chunks;
};


export const addEmbeddings = async (
    textChunks: string[],
    author: string,
    url: string,
    document_name: string,
    collectionName: string
) => {
    const collection = await client.getCollection(collectionName);

    if (!collection) {
        await client.createCollection(collectionName, {
            vectors: {
                size: 4,
                distance: "Cosine"
            },
            optimizers_config: {
                default_segment_number: 2,
            },
            replication_factor: 2
        });
    }

    await Promise.all(
        textChunks.map(async text => {
            const { embedding } = await embed({
                model: mistral.textEmbeddingModel(EMBEDDING_MODEL_NAME),
                value: text
            });

            await client.upsert(collectionName, {
                wait: true,
                points: [{
                    id: generateId(),
                    vector: embedding,
                    payload: {
                        text,
                        citation: {
                            author,
                            url,
                            document_name
                        },
                    }
                }]
            });
        })
    );
};
