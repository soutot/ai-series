import {StreamingTextResponse} from 'ai'
import {NextResponse} from 'next/server'
import {Readable} from 'stream'
import {z} from 'zod'

export async function POST(request: Request) {
  const body = await request.json()
  const bodySchema = z.object({
    prompt: z.string(),
  })

  const {prompt} = bodySchema.parse(body)

  try {
    /* Replace this with actual LLM call */
    const response = new ReadableStream({
      start(controller) {
        controller.enqueue(`Received message: ${prompt}`)
        controller.close()
      },
    })

    return new StreamingTextResponse(response)
  } catch (error) {
    console.log('error', error)
    return new NextResponse(JSON.stringify({error}), {
      status: 500,
      headers: {'content-type': 'application/json'},
    })
  }
}
