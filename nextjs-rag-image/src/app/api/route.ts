import {HNSWLib} from '@langchain/community/vectorstores/hnswlib'
import {BaseMessage} from '@langchain/core/messages'
import {ChatPromptTemplate} from '@langchain/core/prompts'
import {ChatOpenAI, OpenAIEmbeddings} from '@langchain/openai'
import {LangChainStream, StreamingTextResponse} from 'ai'
import {ConversationalRetrievalQAChain} from 'langchain/chains'
import {ChatMessageHistory, ConversationTokenBufferMemory} from 'langchain/memory'
import {NextResponse} from 'next/server'
import {z} from 'zod'

const QA_PROMPT_TEMPLATE = `You are an assistant with limited knowledge, only capable of answering questions based on the provided context, which can be either an image description or a text content.
  Your responses must strictly adhere to the information given in the context; refrain from fabricating answers.
  If a question cannot be answered using the context, respond with "I don't know."
  Politely inform users that you're limited to answering questions related to the provided context.

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
