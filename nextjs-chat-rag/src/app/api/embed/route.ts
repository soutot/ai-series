import {HNSWLib} from '@langchain/community/vectorstores/hnswlib'
import {OpenAIEmbeddings} from '@langchain/openai'
import {RecursiveCharacterTextSplitter} from 'langchain/text_splitter'
import {NextResponse} from 'next/server'

export async function POST(request: Request) {
  const data = request.formData()

  const file: File | null = (await data).get('file') as unknown as File
  if (!file) {
    return NextResponse.json({message: 'Missing file input', success: false})
  }

  const fileContent = await file.text()

  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 100,
  })

  const splitDocs = await textSplitter.createDocuments(fileContent.split('\n'))

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
