local factory = require 'factories.ctf_bounce_factory'

return factory.createLevelApi{
	mapName = 'ctf_bounce',
    episodeLengthSeconds = 1,
    botCount = 7
}