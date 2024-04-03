import {HNSWLib} from '@langchain/community/vectorstores/hnswlib'
import {HumanMessage, MessageContent} from '@langchain/core/messages'
import {ChatPromptTemplate, SystemMessagePromptTemplate} from '@langchain/core/prompts'
import {ChatOpenAI, OpenAIEmbeddings} from '@langchain/openai'
import {LLMChain} from 'langchain/chains'
import {RecursiveCharacterTextSplitter} from 'langchain/text_splitter'
import {NextResponse} from 'next/server'

const SYSTEM_PROMPT = SystemMessagePromptTemplate.fromTemplate(
  `Your task is to generate a detailed and accurate description of the given image. 
  Write a comprehensive description of the image. Include as many relevant details as possible to provide a vivid and clear understanding of what the image portrays.
  Utilize descriptive language to convey the scene depicted in the image. Use adjectives, nouns, verbs, and adverbs effectively to paint a rich picture in the reader's mind.
  Ensure that your description is accurate and faithful to the content of the image. Avoid making assumptions or adding information that is not evident in the image.
  After writing the initial description, review and revise your text to ensure clarity, coherence, and accuracy. Make any necessary adjustments to enhance the quality of your description.
  `
)

export async function POST(request: Request) {
  const data = request.formData()

  const file: File | null = (await data).get('file') as unknown as File
  if (!file) {
    return NextResponse.json({message: 'Missing file input', success: false})
  }

  const buffer = await file.arrayBuffer()
  const base64Image = Buffer.from(buffer).toString('base64')

  const content: Exclude<MessageContent, string> = []

  content.push({
    type: 'image_url',
    image_url: {
      url: `data:image/jpeg;base64,${base64Image}`,
    },
  })

  const visionPrompt = ChatPromptTemplate.fromMessages([
    SYSTEM_PROMPT,
    new HumanMessage({
      content,
    }),
  ])

  const visionLLM = new ChatOpenAI({
    temperature: 0,
    openAIApiKey: process.env.OPENAI_API_KEY,
    modelName: 'gpt-4-vision-preview',
    maxTokens: 4096,
  })

  const visionChain = new LLMChain({
    llm: visionLLM,
    prompt: visionPrompt,
  })

  const response = await visionChain.invoke({})

  const generatedText = response.text

  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 100,
  })

  const splitDocs = await textSplitter.createDocuments(generatedText.split('\n'))

  const embeddings = new OpenAIEmbeddings({
    openAIApiKey: process.env.OPENAI_API_KEY,
  })

  const vectorStore = await HNSWLib.fromDocuments(splitDocs, embeddings)

  await vectorStore.save('vectorstore/rag-store.index')

  return new NextResponse(JSON.stringify({success: true}), {
    status: 200,
    headers: {'content-type': 'application/json'},
  })
}
