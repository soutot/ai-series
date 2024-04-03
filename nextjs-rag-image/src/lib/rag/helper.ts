// import {ChatAnthropic} from '@langchain/anthropic'
import {ChatOllama} from '@langchain/community/chat_models/ollama'
import {HuggingFaceInferenceEmbeddings} from '@langchain/community/embeddings/hf'
import {Chroma} from '@langchain/community/vectorstores/chroma'
import {HNSWLib} from '@langchain/community/vectorstores/hnswlib'
import {Callbacks} from '@langchain/core/callbacks/manager'
import {Document} from '@langchain/core/documents'
import {ChatOpenAI, OpenAIEmbeddings} from '@langchain/openai'
import {existsSync, mkdirSync} from 'fs'
import {RecursiveCharacterTextSplitter} from 'langchain/text_splitter'

import {
  RAG_VECTOR_STORE_PATH,
  VDB_RAG_COLLECTION_NAME,
  HUGGING_FACE_MODEL,
  RAG_PUBLIC_DIRECTORY_PATH,
  DEFAULT_OPENAI_MODEL,
} from '@/app/api/const'

export const getVectorStorePath = (type?: string) => {
  return RAG_VECTOR_STORE_PATH
}

export const getVectorStoreCollectionName = (type?: string) => {
  return VDB_RAG_COLLECTION_NAME
}

export const getLLM = ({model, callbacks}: {callbacks: Callbacks; model: string}) => {
  if (model.includes('gpt-') && process.env.OPENAI_API_KEY) {
    return new ChatOpenAI({
      temperature: 0,
      openAIApiKey: process.env.OPENAI_API_KEY,
      streaming: true,
      modelName: model,
      callbacks,
    })
  }

  // if (model.includes('claude-') && process.env.ANTHROPIC_API_KEY) {
  //   return new ChatAnthropic({
  //     temperature: 0,
  //     modelName: model,
  //     // In Node.js defaults to process.env.ANTHROPIC_API_KEY,
  //     anthropicApiKey: process.env.ANTHROPIC_API_KEY,
  //     streaming: true,
  //   })
  // }

  if (process.env.OLLAMA_API_BASE_URL) {
    return new ChatOllama({
      temperature: 0,
      model: model,
      baseUrl: process.env.OLLAMA_API_BASE_URL,
      callbacks,
    })
  }

  return new ChatOpenAI({
    temperature: 0,
    openAIApiKey: process.env.OPENAI_API_KEY,
    streaming: true,
    modelName: DEFAULT_OPENAI_MODEL,
    callbacks,
  })
}

export const getNonStreamLLM = (props?: {model: string}) => {
  const model = props?.model || DEFAULT_OPENAI_MODEL

  if (model.includes('gpt-') && process.env.OPENAI_API_KEY) {
    return new ChatOpenAI({
      temperature: 0,
      openAIApiKey: process.env.OPENAI_API_KEY,
      modelName: model,
    })
  }

  // if (model.includes('claude-') && process.env.ANTHROPIC_API_KEY) {
  //   return new ChatAnthropic({
  //     temperature: 0,
  //     modelName: model,
  //     // In Node.js defaults to process.env.ANTHROPIC_API_KEY,
  //     anthropicApiKey: process.env.ANTHROPIC_API_KEY,
  //   })
  // }

  if (process.env.OLLAMA_API_BASE_URL) {
    return new ChatOllama({
      temperature: 0,
      model: model,
      baseUrl: process.env.OLLAMA_API_BASE_URL,
    })
  }

  return new ChatOpenAI({
    temperature: 0,
    openAIApiKey: process.env.OPENAI_API_KEY,
    modelName: DEFAULT_OPENAI_MODEL,
  })
}

export const getEmbeddings = () => {
  return process.env.EMBEDDINGS_API_URL
    ? new HuggingFaceInferenceEmbeddings({
        model: HUGGING_FACE_MODEL,
        endpointUrl: process.env.EMBEDDINGS_API_URL,
      })
    : new OpenAIEmbeddings({
        openAIApiKey: process.env.OPENAI_API_KEY,
      })
}

export const getVectorStore = async ({
  vectorStorePath,
  vectorStoreCollectionName,
}: {
  vectorStorePath: string
  vectorStoreCollectionName: string
}) => {
  const embeddings = getEmbeddings()

  return process.env.CHROMADB_URL
    ? await Chroma.fromExistingCollection(new OpenAIEmbeddings(), {
        collectionName: vectorStoreCollectionName,
        url: process.env.CHROMADB_URL,
      })
    : await HNSWLib.load(vectorStorePath, embeddings)
}

export const storeToVectorStore = async ({
  vectorStorePath,
  collectionName,
  docs,
}: {
  vectorStorePath: string
  collectionName: string
  docs: Document<Record<string, any>>[]
}) => {
  const embeddings = getEmbeddings()

  if (process.env.CHROMADB_URL) {
    await Chroma.fromDocuments(docs, embeddings, {
      collectionName,
      url: process.env.CHROMADB_URL,
      collectionMetadata: {
        'hnsw:space': 'cosine',
      },
    })
  } else {
    const vectorStore = await HNSWLib.fromDocuments(docs, embeddings)

    await vectorStore.save(vectorStorePath)
  }
}

function createDirIfNotExists(dir: string) {
  if (!existsSync(dir)) {
    mkdirSync(dir, {recursive: true})
  }
}

export function createWorkingDirs() {
  createDirIfNotExists(RAG_PUBLIC_DIRECTORY_PATH)
}

const getDocsisVersion = (source: string) => {
  if (source.includes('3.0')) {
    return '3.0'
  } else if (source.includes('3.1')) {
    return '3.1'
  } else if (source.includes('4.0')) {
    return '4.0'
  }

  return ''
}

export const getSplitDocs = async (docs: Document<Record<string, any>>[]) => {
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 100,
  })

  const splitDocs = await textSplitter.createDocuments(
    docs.map((doc) => doc.pageContent),
    docs.map((doc) => {
      const version = getDocsisVersion(doc.metadata.source)

      return {
        ...doc.metadata,
        version,
      }
    })
  )

  return splitDocs
}
