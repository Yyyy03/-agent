import {
  LOCAL_AUTH_USERNAME,
  authRequiredResponse,
  isLocalSessionToken,
} from "../local-auth";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  if (!isLocalSessionToken(request.headers.get("X-Mint-Session"))) {
    return authRequiredResponse();
  }

  return Response.json(
    { username: LOCAL_AUTH_USERNAME },
    { headers: { "Cache-Control": "no-store" } },
  );
}
