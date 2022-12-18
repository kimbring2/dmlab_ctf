local factory = require 'factories.ctf_multi_factory'

return factory.createLevelApi{
	mapName = 'ctf_bounce',
    episodeLengthSeconds = 12,
    botCount = 5
}