import { apiClient } from './client';

export interface VoiceTokenRequest {
  room_name: string;
  participant_name: string;
  participant_identity?: string;
  can_publish?: boolean;
  can_subscribe?: boolean;
}

export interface VoiceTokenResponse {
  token: string;
  room_name: string;
  url: string;
}

export const voiceApi = {
  getToken: async (request: VoiceTokenRequest): Promise<VoiceTokenResponse> => {
    return apiClient.post<VoiceTokenResponse>('/api/v1/voice/token', request);
  },
};
