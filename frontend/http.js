// 统一 HTTP + JSON 请求封装：解析 JSON，并把后端错误转换为可读信息。
// 支持浏览器与 Node（node:test）两种环境。
(function (root, factory) {
    const api = factory();
    root.requestJSON = api.requestJSON;
    if (typeof module === 'object' && module.exports) {
        module.exports = api;
    }
}(typeof globalThis !== 'undefined' ? globalThis : this, function () {
    function errorMessage(payload, status) {
        const detail = payload && payload.detail;
        if (typeof detail === 'string') return detail;
        if (detail && typeof detail.message === 'string') return detail.message;
        if (Array.isArray(detail)) {
            const messages = detail.map(item => item && item.msg).filter(Boolean);
            if (messages.length > 0) return messages.join('；');
        }
        if (payload && typeof payload.message === 'string') return payload.message;
        return `请求失败（HTTP ${status}）`;
    }

    async function requestJSON(url, options) {
        const response = await fetch(url, options);
        let payload = null;
        try {
            payload = await response.json();
        } catch (error) {
            if (response.ok) throw new Error('服务端返回了无效的 JSON');
        }
        if (!response.ok) {
            throw new Error(errorMessage(payload, response.status));
        }
        return payload;
    }

    return { requestJSON };
}));