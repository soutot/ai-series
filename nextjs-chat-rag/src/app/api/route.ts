import {HNSWLib} from '@langchain/community/vectorstores/hnswlib'
import {BaseMessage} from '@langchain/core/messages'
import {ChatPromptTemplate} from '@langchain/core/prompts'
import {ChatOpenAI, OpenAIEmbeddings} from '@langchain/openai'
import {LangChainStream, StreamingTextResponse} from 'ai'
import {ConversationalRetrievalQAChain} from 'langchain/chains'
import {ChatMessageHistory, ConversationTokenBufferMemory} from 'langchain/memory'
import {NextResponse} from 'next/server'
import {z} from 'zod'

const QA_PROMPT_TEMPLATE = `You are a good assistant that answers questions. Your knowledge is strictly limited to the following piece of context. Use it to answer the question at the end.
  If the answer can't be found in the context, just say you don't know. *DO NOT* try to make up an answer.
  If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.
  Give a response in the same language as the question.
  
  Context: """"{context}"""

  Question: """{question}"""
  Helpful answer in markdown:`

export async function POST(request: Request) {
  const body = await request.json()
  const bodySchema = z.object({
    prompt: z.string(),
  })

  const {prompt} = bodySchema.parse(body)

  try {
    const embeddings = new OpenAIEmbeddings({
      openAIApiKey: process.env.OPENAI_API_KEY,
    })

    const vectorStore = await HNSWLib.load('vectorstore/rag-store.index', embeddings)

    const retriever = vectorStore.asRetriever()

    const {stream, handlers} = LangChainStream()

    const llm = new ChatOpenAI({
      temperature: 0,
      openAIApiKey: process.env.OPENAI_API_KEY,
      streaming: true,
      modelName: 'gpt-3.5-turbo',
      callbacks: [handlers],
    })

    const chain = ConversationalRetrievalQAChain.fromLLM(llm, retriever, {
      returnSourceDocuments: true,
      qaChainOptions: {
        type: 'stuff',
        prompt: ChatPromptTemplate.fromTemplate(QA_PROMPT_TEMPLATE),
      },
    })

    chain.invoke({question: prompt, chat_history: ''})

    return new StreamingTextResponse(stream)
  } catch (error) {
    console.log('error', error)
    return new NextResponse(JSON.stringify({error}), {
      status: 500,
      headers: {'content-type': 'application/json'},
    })
  }
}
